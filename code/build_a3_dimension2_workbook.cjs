"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { FileBlob, SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const HEADER_FILL = "#1F4E78";
const SUBHEADER_FILL = "#D9EAF7";
const HEADER_FONT = "#FFFFFF";
const WORKBOOK_Z = [0, 0.025, 0.05, 0.075, 0.1, 0.125];

function splitCsv(text) {
  return text.replace(/^\uFEFF/, "").trim().split(/\r?\n/).map((line) => line.split(","));
}

function parseNumericCsv(text, expectedColumns, label) {
  const raw = splitCsv(text);
  const headers = raw[0];
  const rows = raw.slice(1).map((row) => row.map(Number));
  if (headers.length !== expectedColumns || rows.some((row) => row.length !== expectedColumns)) {
    throw new Error(`${label}列数不正确，期望${expectedColumns}列。`);
  }
  if (rows.length < 2 || rows.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error(`${label}包含空值、非数值或数据行不足。`);
  }
  return { headers, rows };
}

function parseTableCsv(text) {
  const raw = splitCsv(text);
  if (raw[0].length !== 6 || raw.slice(1).some((row) => row.length !== 6)) {
    throw new Error("表5中截面CSV应为6列。");
  }
  const rows = raw.slice(1).map((row) => [row[0], ...row.slice(1).map(Number)]);
  if (rows.some((row) => row.slice(1).some((value) => !Number.isFinite(value)))) {
    throw new Error("表5中截面CSV包含非数值水分数据。");
  }
  return { headers: raw[0], rows };
}

function keepWorkbookSample(parsed) {
  const finalTime = parsed.rows.reduce((maximum, row) => Math.max(maximum, row[0]), -Infinity);
  const rows = parsed.rows.filter((row) => (
    WORKBOOK_Z.some((z) => Math.abs(row[2] - z) < 1e-10)
    && (Math.abs(row[0] % 3600) < 1e-10 || row[0] === finalTime)
  ));
  const keptTimes = new Set(rows.map((row) => row[0]));
  if (rows.length !== keptTimes.size * WORKBOOK_Z.length) {
    throw new Error("二维CSV缺少工作簿要求的z截面。完整CSV仍应覆盖0--0.125 m。 ");
  }
  return { headers: parsed.headers, rows };
}

function styleHeader(range) {
  range.format = {
    fill: HEADER_FILL,
    font: { name: "宋体", size: 10, bold: true, color: HEADER_FONT },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  range.format.rowHeight = 30;
}

function styleBody(range) {
  range.format.font = { name: "Arial", size: 9, color: "#202020" };
  range.format.verticalAlignment = "center";
}

function writeChunked(sheet, headers, rows, lastColumn, chunkSize = 3000) {
  sheet.getRange(`A1:${lastColumn}1`).values = [headers];
  for (let offset = 0; offset < rows.length; offset += chunkSize) {
    const chunk = rows.slice(offset, offset + chunkSize);
    const first = offset + 2;
    const last = first + chunk.length - 1;
    sheet.getRange(`A${first}:${lastColumn}${last}`).values = chunk;
  }
}

function configureDataSheet(sheet, headers, rows, lastColumn, kind) {
  writeChunked(sheet, headers, rows, lastColumn);
  const lastRow = rows.length + 1;
  styleHeader(sheet.getRange(`A1:${lastColumn}1`));
  styleBody(sheet.getRange(`A2:${lastColumn}${lastRow}`));
  sheet.getRange(`A2:A${lastRow}`).format.numberFormat = "0";
  if (kind.endsWith("2d")) {
    sheet.getRange(`B2:B${lastRow}`).format.numberFormat = "0.00000000";
    sheet.getRange(`C2:C${lastRow}`).format.numberFormat = "0.000";
    sheet.getRange(`D2:${lastColumn}${lastRow}`).format.numberFormat = kind === "moisture2d" ? "0.0000000000" : "0.000000";
    sheet.getRange("A:A").format.columnWidth = 14;
    sheet.getRange("B:C").format.columnWidth = 13;
    sheet.getRange(`D:${lastColumn}`).format.columnWidth = 12;
    sheet.freezePanes.freezeRows(1);
    sheet.freezePanes.freezeColumns(3);
  } else {
    sheet.getRange(`B2:${lastColumn}${lastRow}`).format.numberFormat = "0.0000000000";
    sheet.getRange("A:A").format.columnWidth = 14;
    sheet.getRange(`B:${lastColumn}`).format.columnWidth = 12;
    sheet.freezePanes.freezeRows(1);
    sheet.freezePanes.freezeColumns(1);
  }
  sheet.showGridLines = false;
  return lastRow;
}

function explanationRows(validation) {
  const scope = validation.model_scope;
  const grid = validation.grid_and_time;
  const drying = validation.drying_time;
  const boundary = validation.constant_boundary_after_4h;
  const balance = validation.whole_product_balances;
  const reference = validation.one_dimensional_reference;
  return [
    ["项目", "数值/设置", "说明"],
    ["模型", "二维轴对称热湿耦合FVM", "固定几何；不含收缩和蒸发潜热"],
    ["计算域", "0≤r≤0.02 m；0≤z≤0.125 m", "只算半圆柱，z=0为中截面对称面"],
    ["对称边界", "r=0，z=0", "零法向通量"],
    ["对流边界", "r=R侧壁，z=0.125 m端面", "侧壁和端面均使用题设hT、hm"],
    ["温度单位", scope.temperature_storage_unit, scope.diffusivity_temperature_unit],
    ["生产网格", `${grid.production.nr}×${grid.production.nz}`, `名义dr=${grid.production.nominal_dr_m} m；名义dz=${grid.production.nominal_dz_m} m；边界加密`],
    ["粗网格", `${grid.coarse.nr}×${grid.coarse.nz}`, `名义dr=${grid.coarse.nominal_dr_m} m；名义dz=${grid.coarse.nominal_dz_m} m；边界加密`],
    ["4 h后温度", boundary.temperature_C, "°C；3--4 h时间加权均值"],
    ["4 h后水分浓度", boundary.moisture_kg_kg, "kg/kg；3--4 h时间加权均值"],
    ["二维烘干时间", drying.production_s, `${drying.production_h} h`],
    ["控制点r", drying.controlling_radius_m, "m"],
    ["控制点z", drying.controlling_z_m, "m"],
    ["终态最大水分", drying.final_maximum_moisture, "严格小于0.15 kg/kg"],
    ["上一时刻最大水分", drying.previous_maximum_moisture, `t=${drying.previous_time_s} s`],
    ["粗细网格时间差", drying.coarse_production_difference_s, "s"],
    ["全场粗细网格最大差", validation.grid_refinement.maximum_sampled_moisture_change_kg_kg, "kg/kg"],
    ["水分守恒相对残差", balance.moisture_relative_residual, "半域通量乘2后按完整药材核算"],
    ["热量平衡相对残差", balance.heat_relative_residual, "非线性表观热容离散核查"],
    ["原一维正式结果", reference && reference.original_production_drying_time_s ? reference.original_production_drying_time_s : "未读取", "s；原模型时间步1/30/1 s"],
    ["一维同时间步结果", reference && reference.matched_time_step_drying_time_s ? reference.matched_time_step_drying_time_s : "未读取", "s；与二维同为30/60/1 s"],
    ["二维-同时间步一维", reference && Number.isFinite(reference.two_dimensional_minus_matched_one_dimensional_s) ? reference.two_dimensional_minus_matched_one_dimensional_s : "未读取", "s；小于二维粗细网格差，未解析出显著烘干时间敏感性"],
    ["二维工作簿采样", `${WORKBOOK_Z.join(", ")} m`, "每小时及最终时刻；完整CSV为每分钟、每0.005 m截面"],
  ];
}

async function main() {
  const [tablePath, midplanePath, moisturePath, temperaturePath, validationPath, outputPath, previewDir] = process.argv.slice(2);
  if (!previewDir) {
    throw new Error("Usage: node build_a3_dimension2_workbook.cjs TABLE MIDPLANE MOISTURE2D TEMPERATURE2D VALIDATION OUTPUT PREVIEW_DIR");
  }
  const validation = JSON.parse(await fs.readFile(validationPath, "utf8"));
  const table = parseTableCsv(await fs.readFile(tablePath, "utf8"));
  const midplane = parseNumericCsv(await fs.readFile(midplanePath, "utf8"), 22, "中截面逐分钟CSV");
  const moisture = keepWorkbookSample(parseNumericCsv(await fs.readFile(moisturePath, "utf8"), 24, "二维水分CSV"));
  const temperature = keepWorkbookSample(parseNumericCsv(await fs.readFile(temperaturePath, "utf8"), 24, "二维温度CSV"));
  const exactFinalMaximum = validation.drying_time.final_maximum_moisture;
  table.rows.at(-1)[1] = exactFinalMaximum;
  midplane.rows.at(-1)[1] = exactFinalMaximum;
  const finalTime = validation.drying_time.production_s;
  const finalMidplane2d = moisture.rows.find((row) => row[0] === finalTime && Math.abs(row[2]) < 1e-12);
  if (!finalMidplane2d) {
    throw new Error("二维水分工作簿采样中找不到最终时刻z=0行。");
  }
  finalMidplane2d[3] = exactFinalMaximum;

  const workbook = Workbook.create();
  const tableSheet = workbook.worksheets.add("表5中截面");
  const midplaneSheet = workbook.worksheets.add("中截面逐分钟");
  const moistureSheet = workbook.worksheets.add("二维水分_每小时");
  const temperatureSheet = workbook.worksheets.add("二维温度_每小时");
  const explanationSheet = workbook.worksheets.add("模型说明");

  writeChunked(tableSheet, table.headers, table.rows, "F");
  styleHeader(tableSheet.getRange("A1:F1"));
  styleBody(tableSheet.getRange(`A2:F${table.rows.length + 1}`));
  tableSheet.getRange(`B2:F${table.rows.length + 1}`).format.numberFormat = "0.0000000000";
  tableSheet.getRange("A:A").format.columnWidth = 24;
  tableSheet.getRange("B:F").format.columnWidth = 14;
  tableSheet.freezePanes.freezeRows(1);
  tableSheet.showGridLines = false;

  const midplaneLastRow = configureDataSheet(midplaneSheet, midplane.headers, midplane.rows, "V", "midplane");
  const moistureLastRow = configureDataSheet(moistureSheet, moisture.headers, moisture.rows, "X", "moisture2d");
  const temperatureLastRow = configureDataSheet(temperatureSheet, temperature.headers, temperature.rows, "X", "temperature2d");

  const notes = explanationRows(validation);
  explanationSheet.getRange(`A1:C${notes.length}`).values = notes;
  styleHeader(explanationSheet.getRange("A1:C1"));
  styleBody(explanationSheet.getRange(`A2:C${notes.length}`));
  explanationSheet.getRange(`A2:A${notes.length}`).format.fill = SUBHEADER_FILL;
  explanationSheet.getRange("A:A").format.columnWidth = 26;
  explanationSheet.getRange("B:B").format.columnWidth = 34;
  explanationSheet.getRange("C:C").format.columnWidth = 56;
  explanationSheet.getRange(`A1:C${notes.length}`).format.wrapText = true;
  explanationSheet.showGridLines = false;

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const reports = [];
  for (const [sheetId, range, rows, cols] of [
    ["表5中截面", `A1:F${Math.min(table.rows.length + 1, 10)}`, 10, 6],
    ["中截面逐分钟", "A1:V5", 5, 22],
    ["中截面逐分钟", `A${midplaneLastRow - 1}:V${midplaneLastRow}`, 2, 22],
    ["二维水分_每小时", "A1:X8", 8, 24],
    ["二维水分_每小时", `A${moistureLastRow - 5}:X${moistureLastRow}`, 6, 24],
    ["二维温度_每小时", "A1:X5", 5, 24],
    ["二维温度_每小时", `A${temperatureLastRow - 2}:X${temperatureLastRow}`, 3, 24],
    ["模型说明", `A1:C${notes.length}`, notes.length, 3],
  ]) {
    const report = await saved.inspect({ kind: "table", sheetId, range, maxChars: 12000, tableMaxRows: rows, tableMaxCols: cols });
    reports.push(report.ndjson);
  }
  const errors = await saved.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 100 },
    summary: "result3_dimension2 formula error scan",
  });
  reports.push(errors.ndjson);

  await fs.mkdir(previewDir, { recursive: true });
  await fs.writeFile(path.join(previewDir, "result3_dimension2_inspection.ndjson"), `${reports.join("\n")}\n`, "utf8");
  for (const [label, sheetName, range] of [
    ["table5", "表5中截面", `A1:F${Math.min(table.rows.length + 1, 12)}`],
    ["moisture2d_top", "二维水分_每小时", "A1:X10"],
    ["moisture2d_bottom", "二维水分_每小时", `A${moistureLastRow - 7}:X${moistureLastRow}`],
    ["model_notes", "模型说明", `A1:C${notes.length}`],
  ]) {
    const preview = await saved.render({ sheetName, range, scale: 1.2, format: "png" });
    await fs.writeFile(path.join(previewDir, `${label}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  console.log(`Verified result3_dimension2.xlsx: midplane=${midplane.rows.length}, moisture2d=${moisture.rows.length}, temperature2d=${temperature.rows.length}.`);
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
