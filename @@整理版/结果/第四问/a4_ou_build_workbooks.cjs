"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { FileBlob, SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const FONT = "Arial";
const HEADER_FILL = "#1F4E78";
const HEADER_FONT = "#FFFFFF";

function parseBoundaryCsv(text) {
  const rows = text
    .replace(/^\uFEFF/, "")
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .map((line) => line.split(",").map(Number));
  if (rows.length !== 4081 || rows.some((row) => row.length !== 4)) {
    throw new Error("Expected 4081 rows and four columns in the OU environment CSV.");
  }
  if (rows.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error("The OU environment CSV contains a non-finite value.");
  }
  rows.forEach((row, index) => {
    if (row[0] !== 14400 + 60 * index) {
      throw new Error(`Unexpected OU time at data row ${index + 1}.`);
    }
  });
  return rows;
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

async function buildResult4(templatePath, payload, outputPath) {
  const headers = payload.headers;
  const rows = payload.rows;
  if (headers.length !== 23 || rows.length < 2 || rows.some((row) => row.length !== 23)) {
    throw new Error("The result4 payload must have 23 columns and at least two rows.");
  }
  for (let index = 0; index < rows.length - 1; index += 1) {
    if (rows[index][0] !== 60 * (index + 1)) {
      throw new Error(`Unexpected result4 time at data row ${index + 1}.`);
    }
  }
  if (rows.at(-1)[0] <= rows.at(-2)[0]) {
    throw new Error("The final drying-time row must follow the preceding row.");
  }

  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
  const sheet = workbook.worksheets.getItem("Sheet1");
  const lastRow = rows.length + 1;
  sheet.getRange(`A1:W${lastRow}`).values = [headers, ...rows];
  sheet.getRange(`A1:W${lastRow}`).format.font = { name: "宋体", size: 11 };
  sheet.getRange("A1:W1").format.font = { name: "宋体", size: 11, bold: true };
  sheet.getRange("A1:W1").format.horizontalAlignment = "center";
  sheet.getRange("A1:W1").format.verticalAlignment = "center";
  sheet.getRange(`A2:A${lastRow}`).format.numberFormat = "0";
  sheet.getRange(`B2:W${lastRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`A2:W${lastRow}`).format.horizontalAlignment = "right";
  sheet.getRange("A:A").format.columnWidth = 24;
  sheet.getRange("B:W").format.columnWidth = 11;
  sheet.getRange("1:1").format.rowHeight = 24;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return { lastRow, rowCount: rows.length };
}

function parameterRows(parameters, validation) {
  const t = parameters.temperature;
  const m = parameters.moisture;
  const realized = validation.realized_ou_boundary_4h_72h;
  return {
    run: [
      ["参数", "数值", "说明"],
      ["方法", parameters.method, "3--4 h稳定段Huber稳健标定；4--72 h联合OU逐分钟生成"],
      ["拟合起点(s)", parameters.fit_interval_s[0], "3 h"],
      ["拟合终点(s)", parameters.fit_interval_s[1], "4 h"],
      ["路径步长(s)", parameters.path_interval_s, "1 min"],
      ["路径终点(s)", parameters.path_end_time_s, "72 h"],
      ["随机种子", String(parameters.random_seed), "固定种子保证结果可复现"],
      ["扰动倍率", parameters.sigma_multiplier, "创新标准差乘数"],
      ["温湿创新相关系数", parameters.robust_innovation_correlation, "Huber截尾创新的同期相关"],
      ["第四问温度输入单位", parameters.question4_temperature_input_unit, parameters.temperature_conversion],
    ],
    fits: [
      [
        "变量", "单位", "样本数", "Huber均衡值", "普通均值", "离散phi",
        "kappa(1/s)", "松弛时间(s)", "半衰期(s)", "创新尺度", "稳态尺度", "phi触及下界",
      ],
      [
        "烘房温度", "°C", t.sample_count, t.equilibrium_mean, t.ordinary_mean,
        t.discrete_phi, t.kappa_per_s, t.relaxation_time_s, t.half_life_s,
        t.innovation_scale, t.stationary_scale, t.phi_at_lower_bound,
      ],
      [
        "烘房水分浓度", "kg/kg", m.sample_count, m.equilibrium_mean, m.ordinary_mean,
        m.discrete_phi, m.kappa_per_s, m.relaxation_time_s, m.half_life_s,
        m.innovation_scale, m.stationary_scale, m.phi_at_lower_bound,
      ],
    ],
    realized: [
      ["变量", "均值", "标准差", "最小值", "最大值", "一阶相关系数"],
      [
        "烘房温度(°C)", realized.temperature_c.mean,
        realized.temperature_c.standard_deviation, realized.temperature_c.minimum,
        realized.temperature_c.maximum, realized.temperature_c.lag1_correlation,
      ],
      [
        "烘房水分浓度(kg/kg)", realized.moisture_kg_kg.mean,
        realized.moisture_kg_kg.standard_deviation, realized.moisture_kg_kg.minimum,
        realized.moisture_kg_kg.maximum, realized.moisture_kg_kg.lag1_correlation,
      ],
    ],
  };
}

async function buildEnvironmentWorkbook(boundaryRows, parameters, validation, outputPath) {
  const workbook = Workbook.create();
  const parameterSheet = workbook.worksheets.add("标定参数");
  const pathSheet = workbook.worksheets.add("逐分钟环境");
  const blocks = parameterRows(parameters, validation);

  parameterSheet.showGridLines = false;
  parameterSheet.getRange("A1:C10").values = blocks.run;
  parameterSheet.getRange("A13:L15").values = blocks.fits;
  parameterSheet.getRange("A18:F20").values = blocks.realized;
  styleHeader(parameterSheet.getRange("A1:C1"));
  styleHeader(parameterSheet.getRange("A13:L13"));
  styleHeader(parameterSheet.getRange("A18:F18"));
  styleBody(parameterSheet.getRange("A2:C10"));
  styleBody(parameterSheet.getRange("A14:L15"));
  styleBody(parameterSheet.getRange("A19:F20"));
  parameterSheet.getRange("B3:B6").format.numberFormat = "0";
  parameterSheet.getRange("B8:B9").format.numberFormat = "0.0000000000";
  parameterSheet.getRange("D14:K15").format.numberFormat = "0.0000000000";
  parameterSheet.getRange("B19:F20").format.numberFormat = "0.0000000000";
  parameterSheet.getRange("A:A").format.columnWidth = 24;
  parameterSheet.getRange("B:B").format.columnWidth = 25;
  parameterSheet.getRange("C:C").format.columnWidth = 52;
  parameterSheet.getRange("D:L").format.columnWidth = 17;
  parameterSheet.getRange("A2:C10").format.wrapText = true;

  const dataRows = boundaryRows.map((row, index) => [
    row[0],
    row[1],
    row[2],
    row[2] + 273.15,
    row[3],
    index === 0 ? "4 h实测衔接值" : "OU模拟",
  ]);
  pathSheet.getRange("A1").write([
    [
      "时间(s)", "时间(h)", "烘房温度(°C)", "烘房温度(K)",
      "烘房水分浓度(kg/kg)", "数据类型",
    ],
    ...dataRows,
  ]);
  styleHeader(pathSheet.getRange("A1:F1"));
  styleBody(pathSheet.getRange(`A2:F${dataRows.length + 1}`));
  pathSheet.getRange(`A2:A${dataRows.length + 1}`).format.numberFormat = "0";
  pathSheet.getRange(`B2:B${dataRows.length + 1}`).format.numberFormat = "0.000000";
  pathSheet.getRange(`C2:D${dataRows.length + 1}`).format.numberFormat = "0.00000000";
  pathSheet.getRange(`E2:E${dataRows.length + 1}`).format.numberFormat = "0.0000000000";
  pathSheet.getRange("A:B").format.columnWidth = 14;
  pathSheet.getRange("C:E").format.columnWidth = 22;
  pathSheet.getRange("F:F").format.columnWidth = 18;
  pathSheet.freezePanes.freezeRows(1);
  pathSheet.showGridLines = false;

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return dataRows.length;
}

async function inspectAndRender(result4Path, environmentPath, previewDir, result4LastRow) {
  await fs.mkdir(previewDir, { recursive: true });
  const result4 = await SpreadsheetFile.importXlsx(await FileBlob.load(result4Path));
  const environment = await SpreadsheetFile.importXlsx(await FileBlob.load(environmentPath));
  const reports = [];
  for (const [workbook, sheetName, range] of [
    [result4, "Sheet1", "A1:W6"],
    [result4, "Sheet1", `A${result4LastRow - 3}:W${result4LastRow}`],
    [environment, "标定参数", "A1:L20"],
    [environment, "逐分钟环境", "A1:F6"],
    [environment, "逐分钟环境", "A4078:F4082"],
  ]) {
    const inspection = await workbook.inspect({
      kind: "table",
      sheetId: sheetName,
      range,
      maxChars: 10000,
      tableMaxRows: 20,
      tableMaxCols: 23,
    });
    reports.push(inspection.ndjson);
  }
  for (const [label, workbook] of [["result4", result4], ["environment", environment]]) {
    const errors = await workbook.inspect({
      kind: "match",
      searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
      options: { useRegex: true, maxResults: 100 },
      summary: `${label} formula error scan`,
    });
    reports.push(errors.ndjson);
  }
  await fs.writeFile(
    path.join(previewDir, "a4_ou_workbooks_inspection.ndjson"),
    `${reports.join("\n")}\n`,
    "utf8",
  );

  for (const [label, workbook, sheetName, range] of [
    ["result4_top", result4, "Sheet1", "A1:W8"],
    ["result4_bottom", result4, "Sheet1", `A${result4LastRow - 6}:W${result4LastRow}`],
    ["environment_parameters", environment, "标定参数", "A1:L20"],
    ["environment_path_top", environment, "逐分钟环境", "A1:F8"],
    ["environment_path_bottom", environment, "逐分钟环境", "A4076:F4082"],
  ]) {
    const preview = await workbook.render({ sheetName, range, scale: 1.25, format: "png" });
    await fs.writeFile(
      path.join(previewDir, `${label}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}

async function main() {
  const [
    templatePath, payloadPath, boundaryCsvPath, parametersPath, validationPath,
    result4Path, environmentPath, previewDir,
  ] = process.argv.slice(2);
  if (!previewDir) {
    throw new Error(
      "Usage: node a4_ou_build_workbooks.cjs TEMPLATE PAYLOAD BOUNDARY_CSV PARAMETERS VALIDATION RESULT4_XLSX ENV_XLSX PREVIEW_DIR",
    );
  }
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const boundaryRows = parseBoundaryCsv(await fs.readFile(boundaryCsvPath, "utf8"));
  const parameters = JSON.parse(await fs.readFile(parametersPath, "utf8"));
  const validation = JSON.parse(await fs.readFile(validationPath, "utf8"));

  const result4Meta = await buildResult4(templatePath, payload, result4Path);
  const environmentRows = await buildEnvironmentWorkbook(
    boundaryRows, parameters, validation, environmentPath,
  );
  await inspectAndRender(result4Path, environmentPath, previewDir, result4Meta.lastRow);
  console.log(
    `Verified result4.xlsx (${result4Meta.rowCount} data rows) and ` +
    `ou_environment_parameters.xlsx (${environmentRows} environment rows).`,
  );
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
