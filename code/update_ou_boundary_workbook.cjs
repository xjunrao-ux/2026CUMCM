"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { FileBlob, SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const MAX_EXCEL_ROWS = 1048576;
const MAX_STORED_RUNS = 200;
const FONT = "Arial";
const HEADER_FILL = "#1F4E78";
const HEADER_FONT = "#FFFFFF";

function parseNumericCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const rows = lines.slice(1).map((line) => line.split(",").map(Number));
  if (rows.length !== 4081 || rows.some((row) => row.length !== 4)) {
    throw new Error("Expected 4081 OU boundary rows and four numeric columns.");
  }
  if (rows.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error("The OU boundary CSV contains a non-finite value.");
  }
  for (let index = 0; index < rows.length; index += 1) {
    const expectedTime = 14400 + 60 * index;
    if (rows[index][0] !== expectedTime) {
      throw new Error(`Unexpected OU time at CSV data row ${index + 1}.`);
    }
  }
  return rows;
}

function getSheet(workbook, name) {
  return workbook.worksheets.items.find((sheet) => sheet.name === name) || null;
}

function getOrAddSheet(workbook, name) {
  return getSheet(workbook, name) || workbook.worksheets.add(name);
}

function existingRows(sheet) {
  if (!sheet) return [];
  const used = sheet.getUsedRange(true);
  if (!used) return [];
  const values = used.values || [];
  return values.length > 1 ? values.slice(1).filter((row) => row[0] != null) : [];
}

function clearSheet(sheet) {
  const used = sheet.getUsedRange();
  if (used) used.clear({ applyTo: "all" });
  sheet.deleteAllDrawings();
}

function styleHeader(range) {
  range.format = {
    fill: HEADER_FILL,
    font: { name: FONT, size: 10, bold: true, color: HEADER_FONT },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
  };
  range.format.rowHeight = 30;
}

function styleBody(range) {
  range.format.font = { name: FONT, size: 10, color: "#202020" };
  range.format.verticalAlignment = "center";
}

function addLineChart(sheet, categoryRange, valueRange, title, positionStart, positionEnd, numberFormat) {
  const chart = sheet.charts.add("line", [categoryRange, valueRange]);
  chart.title = title;
  chart.titleTextStyle.fontSize = 11;
  chart.titleTextStyle.typeface = FONT;
  chart.hasLegend = false;
  chart.xAxis = {
    axisType: "textAxis",
    numberFormatCode: "0",
    numberFormatSourceLinked: false,
    textStyle: { typeface: FONT, fontSize: 9 },
  };
  chart.yAxis = {
    numberFormatCode: numberFormat,
    numberFormatSourceLinked: false,
    textStyle: { typeface: FONT, fontSize: 9 },
  };
  chart.setPosition(positionStart, positionEnd);
  return chart;
}

function addColumnChart(sheet, categoryRange, valueRange, title, positionStart, positionEnd, numberFormat) {
  const chart = sheet.charts.add("column", [categoryRange, valueRange]);
  chart.title = title;
  chart.titleTextStyle.fontSize = 11;
  chart.titleTextStyle.typeface = FONT;
  chart.hasLegend = false;
  chart.xAxis = {
    axisType: "textAxis",
    textStyle: { typeface: FONT, fontSize: 9 },
  };
  chart.yAxis = {
    numberFormatCode: numberFormat,
    numberFormatSourceLinked: false,
    textStyle: { typeface: FONT, fontSize: 9 },
  };
  chart.setPosition(positionStart, positionEnd);
  return chart;
}

async function main() {
  const [boundaryCsvPath, parametersPath, validationPath, outputPath, previewDir] =
    process.argv.slice(2);
  if (!boundaryCsvPath || !parametersPath || !validationPath || !outputPath || !previewDir) {
    throw new Error(
      "Usage: node update_ou_boundary_workbook.cjs BOUNDARY_CSV PARAMETERS_JSON VALIDATION_JSON OUTPUT_XLSX PREVIEW_DIR",
    );
  }

  const boundaryRows = parseNumericCsv(await fs.readFile(boundaryCsvPath, "utf8"));
  const parameters = JSON.parse(await fs.readFile(parametersPath, "utf8"));
  const validation = JSON.parse(await fs.readFile(validationPath, "utf8"));
  const outputExists = await fs.access(outputPath).then(() => true).catch(() => false);
  const workbook = outputExists
    ? await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath))
    : Workbook.create();

  const summarySheet = getOrAddSheet(workbook, "运行汇总");
  const currentSheet = getOrAddSheet(workbook, "当前路径");
  const historySheet = getOrAddSheet(workbook, "路径历史");
  const previousSummary = existingRows(summarySheet);
  const previousHistory = existingRows(historySheet);
  if (previousSummary.length >= MAX_STORED_RUNS) {
    throw new Error(`Workbook already contains ${MAX_STORED_RUNS} runs; archive it before appending.`);
  }
  if (previousHistory.length + boundaryRows.length + 1 > MAX_EXCEL_ROWS) {
    throw new Error("Appending this run would exceed Excel's worksheet row limit.");
  }

  const runIndex = previousSummary.length + 1;
  const runId = `R${String(runIndex).padStart(3, "0")}_seed_${parameters.random_seed}`;
  const runUtc = new Date().toISOString();
  const realized = validation.realized_ou_boundary_4h_72h;
  const reference = validation.comparison_with_constant_mean_model || {};
  const summaryHeader = [
    "运行序号",
    "运行ID",
    "运行时间(UTC)",
    "随机种子",
    "扰动倍率",
    "温度稳态均值(°C)",
    "水分稳态均值(kg/kg)",
    "温度φ",
    "水分φ",
    "创新相关系数",
    "预测烘干时间(h)",
    "相对恒均值变化(min)",
    "路径温度均值(°C)",
    "路径温度标准差(°C)",
    "路径温度最小值(°C)",
    "路径温度最大值(°C)",
    "路径水分均值(kg/kg)",
    "路径水分标准差(kg/kg)",
    "路径水分最小值(kg/kg)",
    "路径水分最大值(kg/kg)",
  ];
  const summaryRow = [
    runIndex,
    runId,
    runUtc,
    String(parameters.random_seed),
    parameters.sigma_multiplier,
    parameters.temperature.equilibrium_mean,
    parameters.moisture.equilibrium_mean,
    parameters.temperature.discrete_phi,
    parameters.moisture.discrete_phi,
    parameters.robust_innovation_correlation,
    validation.drying_time.production_h,
    reference.delta_ou_minus_mean_min ?? null,
    realized.temperature_c.mean,
    realized.temperature_c.standard_deviation,
    realized.temperature_c.minimum,
    realized.temperature_c.maximum,
    realized.moisture_kg_kg.mean,
    realized.moisture_kg_kg.standard_deviation,
    realized.moisture_kg_kg.minimum,
    realized.moisture_kg_kg.maximum,
  ];
  const summaryRows = [...previousSummary, summaryRow];

  const historyHeader = [
    "运行序号",
    "运行ID",
    "随机种子",
    "时间(s)",
    "时间(h)",
    "烘房温度(°C)",
    "烘房水分浓度(kg/kg)",
  ];
  const appendedHistory = boundaryRows.map((row) => [
    runIndex,
    runId,
    String(parameters.random_seed),
    row[0],
    row[1],
    row[2],
    row[3],
  ]);
  const historyRows = [...previousHistory, ...appendedHistory];

  clearSheet(summarySheet);
  clearSheet(currentSheet);
  clearSheet(historySheet);

  summarySheet.getRange("A1").write([summaryHeader, ...summaryRows]);
  styleHeader(summarySheet.getRange("A1:T1"));
  styleBody(summarySheet.getRange(`A2:T${summaryRows.length + 1}`));
  summarySheet.getRange(`A2:A${summaryRows.length + 1}`).format.numberFormat = "0";
  summarySheet.getRange(`C2:C${summaryRows.length + 1}`).format.numberFormat =
    "yyyy-mm-dd hh:mm:ss";
  summarySheet.getRange(`D2:D${summaryRows.length + 1}`).format.numberFormat = "@";
  summarySheet.getRange(`E2:E${summaryRows.length + 1}`).format.numberFormat = "0.00";
  summarySheet.getRange(`F2:F${summaryRows.length + 1}`).format.numberFormat = "0.000000";
  summarySheet.getRange(`G2:G${summaryRows.length + 1}`).format.numberFormat = "0.0000000000";
  summarySheet.getRange(`H2:J${summaryRows.length + 1}`).format.numberFormat = "0.000000";
  summarySheet.getRange(`K2:L${summaryRows.length + 1}`).format.numberFormat = "0.000000";
  summarySheet.getRange(`M2:P${summaryRows.length + 1}`).format.numberFormat = "0.000000";
  summarySheet.getRange(`Q2:T${summaryRows.length + 1}`).format.numberFormat = "0.0000000000";
  summarySheet.getRange("A:A").format.columnWidth = 10;
  summarySheet.getRange("B:B").format.columnWidth = 25;
  summarySheet.getRange("C:C").format.columnWidth = 23;
  summarySheet.getRange("D:T").format.columnWidth = 17;
  summarySheet.freezePanes.freezeRows(1);
  summarySheet.showGridLines = false;

  const summaryLastRow = summaryRows.length + 1;
  addColumnChart(
    summarySheet,
    summarySheet.getRange(`A1:A${summaryLastRow}`),
    summarySheet.getRange(`K1:K${summaryLastRow}`),
    "各次运行预测烘干时间 (h)",
    "V2",
    "AD17",
    "0.000",
  );
  addColumnChart(
    summarySheet,
    summarySheet.getRange(`A1:A${summaryLastRow}`),
    summarySheet.getRange(`M1:M${summaryLastRow}`),
    "各次路径平均温度 (°C)",
    "V19",
    "AD34",
    "0.000",
  );
  addColumnChart(
    summarySheet,
    summarySheet.getRange(`A1:A${summaryLastRow}`),
    summarySheet.getRange(`Q1:Q${summaryLastRow}`),
    "各次路径平均水分浓度 (kg/kg)",
    "V36",
    "AD51",
    "0.000000",
  );

  const currentHeader = [
    "时间(s)",
    "时间(h)",
    "烘房温度(°C)",
    "烘房水分浓度(kg/kg)",
  ];
  currentSheet.getRange("A1").write([currentHeader, ...boundaryRows]);
  styleHeader(currentSheet.getRange("A1:D1"));
  styleBody(currentSheet.getRange(`A2:D${boundaryRows.length + 1}`));
  currentSheet.getRange(`A2:A${boundaryRows.length + 1}`).format.numberFormat = "0";
  currentSheet.getRange(`B2:B${boundaryRows.length + 1}`).format.numberFormat = "0.000000";
  currentSheet.getRange(`C2:C${boundaryRows.length + 1}`).format.numberFormat = "0.000000";
  currentSheet.getRange(`D2:D${boundaryRows.length + 1}`).format.numberFormat = "0.0000000000";
  currentSheet.getRange("A:D").format.columnWidth = 21;

  const hourlyRows = boundaryRows.filter(
    (row, index) => index % 60 === 0 || index === boundaryRows.length - 1,
  );
  currentSheet.getRange("F1").write([
    ["趋势时间(h)", "温度(°C)", "水分浓度(kg/kg)"],
    ...hourlyRows.map((row) => [row[1], row[2], row[3]]),
  ]);
  styleHeader(currentSheet.getRange("F1:H1"));
  styleBody(currentSheet.getRange(`F2:H${hourlyRows.length + 1}`));
  currentSheet.getRange(`F2:F${hourlyRows.length + 1}`).format.numberFormat = "0";
  currentSheet.getRange(`G2:G${hourlyRows.length + 1}`).format.numberFormat = "0.000000";
  currentSheet.getRange(`H2:H${hourlyRows.length + 1}`).format.numberFormat = "0.0000000000";
  currentSheet.getRange("F:H").format.columnWidth = 21;
  currentSheet.freezePanes.freezeRows(1);
  currentSheet.showGridLines = false;

  const hourlyLastRow = hourlyRows.length + 1;
  addLineChart(
    currentSheet,
    currentSheet.getRange(`F1:F${hourlyLastRow}`),
    currentSheet.getRange(`G1:G${hourlyLastRow}`),
    `当前路径温度趋势 (${runId})`,
    "J2",
    "R18",
    "0.00",
  );
  addLineChart(
    currentSheet,
    currentSheet.getRange(`F1:F${hourlyLastRow}`),
    currentSheet.getRange(`H1:H${hourlyLastRow}`),
    `当前路径水分浓度趋势 (${runId})`,
    "J20",
    "R36",
    "0.00000",
  );

  historySheet.getRange("A1").write([historyHeader, ...historyRows]);
  styleHeader(historySheet.getRange("A1:G1"));
  styleBody(historySheet.getRange(`A2:G${historyRows.length + 1}`));
  historySheet.getRange(`A2:A${historyRows.length + 1}`).format.numberFormat = "0";
  historySheet.getRange(`C2:C${historyRows.length + 1}`).format.numberFormat = "@";
  historySheet.getRange(`D2:D${historyRows.length + 1}`).format.numberFormat = "0";
  historySheet.getRange(`E2:E${historyRows.length + 1}`).format.numberFormat = "0.000000";
  historySheet.getRange(`F2:F${historyRows.length + 1}`).format.numberFormat = "0.000000";
  historySheet.getRange(`G2:G${historyRows.length + 1}`).format.numberFormat = "0.0000000000";
  historySheet.getRange("A:A").format.columnWidth = 10;
  historySheet.getRange("B:B").format.columnWidth = 25;
  historySheet.getRange("C:G").format.columnWidth = 21;
  historySheet.freezePanes.freezeRows(1);

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const summaryInspection = await saved.inspect({
    kind: "table,drawing",
    sheetId: "运行汇总",
    range: `A1:T${summaryLastRow}`,
    maxChars: 8000,
    tableMaxRows: 8,
    tableMaxCols: 20,
  });
  const currentInspection = await saved.inspect({
    kind: "table,drawing",
    sheetId: "当前路径",
    range: "A1:H8",
    maxChars: 8000,
    tableMaxRows: 8,
    tableMaxCols: 8,
  });
  const historyInspection = await saved.inspect({
    kind: "table",
    sheetId: "路径历史",
    range: `A${historyRows.length - 1}:G${historyRows.length + 1}`,
    maxChars: 5000,
    tableMaxRows: 3,
    tableMaxCols: 7,
  });
  const errors = await saved.inspect({
    kind: "match",
    searchTerm:
      "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });

  await fs.mkdir(previewDir, { recursive: true });
  await fs.writeFile(
    path.join(previewDir, "ou_boundary_runs_inspection.ndjson"),
    `${summaryInspection.ndjson}\n${currentInspection.ndjson}\n${historyInspection.ndjson}\n${errors.ndjson}\n`,
    "utf8",
  );
  for (const [sheetName, range, fileName] of [
    ["运行汇总", "A1:AD51", "summary.png"],
    ["当前路径", "A1:R36", "current.png"],
    ["路径历史", "A1:G12", "history.png"],
  ]) {
    const preview = await saved.render({
      sheetName,
      range,
      scale: 1,
      format: "png",
    });
    await fs.writeFile(
      path.join(previewDir, fileName),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
  console.log(
    `${outputExists ? "Updated" : "Created"} OU workbook: ${runId}; ` +
      `${summaryRows.length} runs, ${historyRows.length} history rows.`,
  );
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
