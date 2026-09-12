"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { FileBlob, SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const HEADER_FILL = "#1F4E78";
const SECTION_FILL = "#D9EAF7";
const WORKBOOK_Z = [0, 0.025, 0.05, 0.075, 0.1, 0.125];

function splitCsv(text) {
  return text.replace(/^\uFEFF/, "").trim().split(/\r?\n/).map((line) => line.split(","));
}

function numericOrNull(value) {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`非数值字段: ${value}`);
  return parsed;
}

function parseDataCsv(text, expectedColumns, label) {
  const raw = splitCsv(text);
  if (raw[0].length !== expectedColumns || raw.slice(1).some((row) => row.length !== expectedColumns)) {
    throw new Error(`${label}列数不正确，期望${expectedColumns}列。`);
  }
  const rows = raw.slice(1).map((row) => row.map(numericOrNull));
  if (rows.length < 2 || rows.some((row) => row[0] === null)) {
    throw new Error(`${label}缺少有效数据行或时间。`);
  }
  return { headers: raw[0], rows };
}

function parseTableCsv(text) {
  const raw = splitCsv(text);
  if (raw[0].length !== 7 || raw.slice(1).some((row) => row.length !== 7)) {
    throw new Error("表6中截面CSV应为7列。");
  }
  return {
    headers: raw[0],
    rows: raw.slice(1).map((row) => [row[0], ...row.slice(1).map(numericOrNull)]),
  };
}

function keepHourlySections(parsed) {
  const finalTime = parsed.rows.reduce((maximum, row) => Math.max(maximum, row[0]), -Infinity);
  const rows = parsed.rows.filter((row) => (
    WORKBOOK_Z.some((z) => Math.abs(row[3] - z) < 1e-10)
    && (Math.abs(row[0] % 3600) < 1e-10 || row[0] === finalTime)
  ));
  const keptTimes = new Set(rows.map((row) => row[0]));
  if (rows.length !== keptTimes.size * WORKBOOK_Z.length) {
    throw new Error("二维CSV缺少工作簿要求的时间或z截面。");
  }
  return { headers: parsed.headers, rows };
}

function styleHeader(range) {
  range.format = {
    fill: HEADER_FILL,
    font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
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

function configureMidplane(sheet, parsed) {
  // Preserve original result4 layout: time + 21 fixed radii + moving surface.
  const headers = [parsed.headers[0], ...parsed.headers.slice(2)];
  const rows = parsed.rows.map((row) => [row[0], ...row.slice(2)]);
  writeChunked(sheet, headers, rows, "W");
  const lastRow = rows.length + 1;
  styleHeader(sheet.getRange("A1:W1"));
  styleBody(sheet.getRange(`A2:W${lastRow}`));
  sheet.getRange(`A2:A${lastRow}`).format.numberFormat = "0";
  sheet.getRange(`B2:W${lastRow}`).format.numberFormat = "0.0000000000";
  sheet.getRange("A:A").format.columnWidth = 24;
  sheet.getRange("B:W").format.columnWidth = 12;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
  sheet.showGridLines = false;
  return lastRow;
}

function configure2d(sheet, parsed, isMoisture) {
  writeChunked(sheet, parsed.headers, parsed.rows, "Z");
  const lastRow = parsed.rows.length + 1;
  styleHeader(sheet.getRange("A1:Z1"));
  styleBody(sheet.getRange(`A2:Z${lastRow}`));
  sheet.getRange(`A2:A${lastRow}`).format.numberFormat = "0";
  sheet.getRange(`B2:D${lastRow}`).format.numberFormat = "0.00000000";
  sheet.getRange(`E2:Z${lastRow}`).format.numberFormat = isMoisture ? "0.0000000000" : "0.000000";
  sheet.getRange("A:A").format.columnWidth = 14;
  sheet.getRange("B:D").format.columnWidth = 13;
  sheet.getRange("E:Z").format.columnWidth = 12;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(4);
  sheet.showGridLines = false;
  return lastRow;
}

function noteRows(validation) {
  const model = validation.model_scope;
  const grid = validation.grid_and_time;
  const boundary = validation.constant_boundary_after_4h;
  const radius = validation.radius;
  const drying = validation.drying_time;
  const compare = validation.one_dimensional_comparison;
  const refinement = validation.grid_refinement;
  const balance = validation.whole_product_balances;
  return [
    ["项目", "数值/设置", "说明"],
    ["模型", "二维轴对称收缩FVM", "沿用问题四主模型，不含蒸发潜热"],
    ["计算域", "0≤xi=r/R(t)≤1；0≤z≤0.125 m", "只计算半圆柱"],
    ["半径变化", model.radial_shrinkage, "附件2分段线性插值"],
    ["轴向变化", model.axial_shrinkage, "长度固定0.25 m"],
    ["材料经验公式", model.constitutive_laws, "附件4公式未修改"],
    ["温度单位", model.temperature_state_unit, "扩散系数中的T为开尔文"],
    ["潜热/蒸发源", `${model.latent_heat} / ${model.evaporation_source}`, "不进入方程"],
    ["生产网格", `${grid.production.nr}×${grid.production.nz}`, `名义初始dr=${grid.production.nominal_initial_dr_m} m；dz=${grid.production.nominal_dz_m} m`],
    ["粗网格", `${grid.coarse.nr}×${grid.coarse.nz}`, "用于空间收敛核查"],
    ["时间步", grid.time_steps_s.join(" / "), "s；实测段/恒定段/阈值段"],
    ["4 h后温度", boundary.temperature_K, "K；3--4 h时间加权均值"],
    ["4 h后水分浓度", boundary.moisture_kg_kg, "kg/kg；3--4 h时间加权均值"],
    ["初始半径", radius.initial_cm, "cm"],
    ["终止半径", radius.drying_time_cm, "cm"],
    ["二维烘干时间", drying.production_s, `${drying.production_h} h`],
    ["上一时刻最大水分", drying.previous_maximum_moisture, `t=${drying.previous_time_s} s`],
    ["终态最大水分", drying.final_maximum_moisture, "严格小于0.15 kg/kg"],
    ["控制点", `r=${drying.controlling_radius_m} m；z=${drying.controlling_z_m} m`, "全二维最大值位置"],
    ["原一维正式结果", compare.original_main_model_s, "s；主问题四高分辨率结果"],
    ["一维同网格时间步", compare.matched_grid_time_step_s, "s；用于隔离端面效应"],
    ["二维-同条件一维", compare.two_dimensional_minus_matched_one_dimensional_s, "s；负值表示端面加快干燥"],
    ["粗细网格时间差", drying.coarse_production_difference_s, "s"],
    ["粗细网格全场最大差", refinement.maximum_sampled_moisture_change_kg_kg, "kg/kg"],
    ["水分守恒相对残差", balance.moisture_relative_residual, "完整药材尺度"],
    ["热量平衡相对残差", balance.heat_relative_residual, "完整药材尺度"],
    ["二维工作簿采样", `${WORKBOOK_Z.join(", ")} m`, "每小时及最终时刻；完整CSV为每分钟、每0.005 m"],
  ];
}

async function main() {
  const [tablePath, midplanePath, moisturePath, temperaturePath, validationPath, outputPath, previewDir] = process.argv.slice(2);
  if (!previewDir) throw new Error("Usage: node build_a4_dimension2_workbook.cjs TABLE MIDPLANE MOISTURE TEMPERATURE VALIDATION OUTPUT PREVIEW");

  const validation = JSON.parse(await fs.readFile(validationPath, "utf8"));
  const table = parseTableCsv(await fs.readFile(tablePath, "utf8"));
  const midplane = parseDataCsv(await fs.readFile(midplanePath, "utf8"), 24, "中截面逐分钟CSV");
  const moisture = keepHourlySections(parseDataCsv(await fs.readFile(moisturePath, "utf8"), 26, "二维水分CSV"));
  const temperature = keepHourlySections(parseDataCsv(await fs.readFile(temperaturePath, "utf8"), 26, "二维温度CSV"));

  const exactMaximum = validation.drying_time.final_maximum_moisture;
  table.rows.at(-1)[1] = exactMaximum;
  midplane.rows.at(-1)[2] = exactMaximum;
  const finalTime = validation.drying_time.production_s;
  const finalMiddle = moisture.rows.find((row) => row[0] === finalTime && Math.abs(row[3]) < 1e-12);
  if (!finalMiddle) throw new Error("二维水分中找不到最终时刻z=0行。");
  finalMiddle[4] = exactMaximum;

  const workbook = Workbook.create();
  const tableSheet = workbook.worksheets.add("表6中截面");
  const resultSheet = workbook.worksheets.add("result4中截面逐分钟");
  const moistureSheet = workbook.worksheets.add("二维水分_每小时");
  const temperatureSheet = workbook.worksheets.add("二维温度K_每小时");
  const noteSheet = workbook.worksheets.add("模型说明");

  writeChunked(tableSheet, table.headers, table.rows, "G");
  styleHeader(tableSheet.getRange("A1:G1"));
  styleBody(tableSheet.getRange(`A2:G${table.rows.length + 1}`));
  tableSheet.getRange(`B2:G${table.rows.length + 1}`).format.numberFormat = "0.0000000000";
  tableSheet.getRange("A:A").format.columnWidth = 24;
  tableSheet.getRange("B:G").format.columnWidth = 14;
  tableSheet.freezePanes.freezeRows(1);
  tableSheet.showGridLines = false;

  const resultLastRow = configureMidplane(resultSheet, midplane);
  const moistureLastRow = configure2d(moistureSheet, moisture, true);
  const temperatureLastRow = configure2d(temperatureSheet, temperature, false);

  const notes = noteRows(validation);
  noteSheet.getRange(`A1:C${notes.length}`).values = notes;
  styleHeader(noteSheet.getRange("A1:C1"));
  styleBody(noteSheet.getRange(`A2:C${notes.length}`));
  noteSheet.getRange(`A2:A${notes.length}`).format.fill = SECTION_FILL;
  noteSheet.getRange("A:A").format.columnWidth = 27;
  noteSheet.getRange("B:B").format.columnWidth = 42;
  noteSheet.getRange("C:C").format.columnWidth = 52;
  noteSheet.getRange(`A1:C${notes.length}`).format.wrapText = true;
  noteSheet.showGridLines = false;

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const reports = [];
  for (const [sheetId, range, rows, cols] of [
    ["表6中截面", `A1:G${table.rows.length + 1}`, table.rows.length + 1, 7],
    ["result4中截面逐分钟", "A1:W5", 5, 23],
    ["result4中截面逐分钟", `A${resultLastRow - 1}:W${resultLastRow}`, 2, 23],
    ["二维水分_每小时", "A1:Z8", 8, 26],
    ["二维水分_每小时", `A${moistureLastRow - 5}:Z${moistureLastRow}`, 6, 26],
    ["二维温度K_每小时", "A1:Z5", 5, 26],
    ["二维温度K_每小时", `A${temperatureLastRow - 2}:Z${temperatureLastRow}`, 3, 26],
    ["模型说明", `A1:C${notes.length}`, notes.length, 3],
  ]) {
    const report = await saved.inspect({ kind: "table", sheetId, range, maxChars: 14000, tableMaxRows: rows, tableMaxCols: cols });
    reports.push(report.ndjson);
  }
  const errors = await saved.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 100 },
    summary: "result4_dimension2 formula error scan",
  });
  reports.push(errors.ndjson);
  await fs.mkdir(previewDir, { recursive: true });
  await fs.writeFile(path.join(previewDir, "result4_dimension2_inspection.ndjson"), `${reports.join("\n")}\n`, "utf8");
  for (const [label, sheetName, range] of [
    ["table6", "表6中截面", `A1:G${table.rows.length + 1}`],
    ["result4_top", "result4中截面逐分钟", "A1:W8"],
    ["result4_bottom", "result4中截面逐分钟", `A${resultLastRow - 6}:W${resultLastRow}`],
    ["moisture2d_top", "二维水分_每小时", "A1:Z10"],
    ["moisture2d_bottom", "二维水分_每小时", `A${moistureLastRow - 7}:Z${moistureLastRow}`],
    ["temperature2d_top", "二维温度K_每小时", "A1:Z8"],
    ["notes", "模型说明", `A1:C${notes.length}`],
  ]) {
    const preview = await saved.render({ sheetName, range, scale: 1.2, format: "png" });
    await fs.writeFile(path.join(previewDir, `${label}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
  console.log(`Verified result4_dimension2.xlsx: result4=${midplane.rows.length}, moisture2d=${moisture.rows.length}, temperature2d=${temperature.rows.length}.`);
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
