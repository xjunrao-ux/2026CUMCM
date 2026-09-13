"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { FileBlob, SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const FONT = "Arial";
const HEADER_FILL = "#1F4E78";
const HEADER_FONT = "#FFFFFF";

function parseInputCsv(text) {
  const rows = text
    .replace(/^\uFEFF/, "")
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .map((line) => {
      const parts = line.split(",");
      return [...parts.slice(0, 11).map(Number), parts[11]];
    });
  if (rows.length !== 4081 || rows.some((row) => row.length !== 12)) {
    throw new Error("Expected 4081 rows and 12 columns in the 4--72 h input CSV.");
  }
  if (rows.some((row) => row.slice(0, 11).some((value) => !Number.isFinite(value)))) {
    throw new Error("The 4--72 h input CSV contains a non-finite numeric value.");
  }
  rows.forEach((row, index) => {
    if (row[0] !== 14400 + 60 * index) {
      throw new Error(`Unexpected time at 4--72 h input row ${index + 1}.`);
    }
    const reconstructed = row[8] + row[10] * (row[0] - row[6]);
    if (Math.abs(reconstructed - row[2]) > 1e-9) {
      throw new Error(`Piecewise-linear radius mismatch at input row ${index + 1}.`);
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
  const { headers, rows } = payload;
  if (
    headers.length !== 23
    || rows.length < 2
    || rows.some((row) => row.length !== 23)
  ) {
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

  const workbook = await SpreadsheetFile.importXlsx(
    await FileBlob.load(templatePath),
  );
  const sheet = workbook.worksheets.getItem("Sheet1");
  const lastRow = rows.length + 1;
  sheet.getRange(`A1:W${lastRow}`).values = [headers, ...rows];
  sheet.getRange(`A1:W${lastRow}`).format.font = { name: "宋体", size: 11 };
  sheet.getRange("A1:W1").format.font = { name: "宋体", size: 11, bold: true };
  sheet.getRange("A1:W1").format.horizontalAlignment = "center";
  sheet.getRange("A1:W1").format.verticalAlignment = "center";
  sheet.getRange(`A2:A${lastRow}`).format.numberFormat = "0";
  // 保留8位，保证最终严格小于0.15的状态不会被四舍五入掩盖。
  sheet.getRange(`B2:W${lastRow}`).format.numberFormat = "0.00000000";
  sheet.getRange(`A2:W${lastRow}`).format.horizontalAlignment = "right";
  sheet.getRange("A:A").format.columnWidth = 24;
  sheet.getRange("B:W").format.columnWidth = 15;
  sheet.getRange("1:1").format.rowHeight = 24;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return { lastRow, rowCount: rows.length };
}

async function buildInputsWorkbook(inputRows, attachmentRows, validation, outputPath) {
  const workbook = Workbook.create();
  const minuteSheet = workbook.worksheets.add("逐分钟输入");
  const sourceSheet = workbook.worksheets.add("附件2节点");
  const minuteLastRow = inputRows.length + 1;

  minuteSheet.getRange("A1:N1").values = [[
    "时间(s)",
    "时间(h)",
    "模型半径(cm)",
    "烘房温度(°C)",
    "烘房温度(K)",
    "烘房水分浓度(kg/kg)",
    "分段起点(s)",
    "分段终点(s)",
    "起点半径(cm)",
    "终点半径(cm)",
    "斜率(cm/s)",
    "数据类型",
    "Excel复算半径(cm)",
    "复算差值(cm)",
  ]];
  minuteSheet.getRange(`A2:L${minuteLastRow}`).values = inputRows;
  minuteSheet.getRange("M2").formulas = [["=I2+K2*(A2-G2)"]];
  minuteSheet.getRange(`M2:M${minuteLastRow}`).fillDown();
  minuteSheet.getRange("N2").formulas = [["=C2-M2"]];
  minuteSheet.getRange(`N2:N${minuteLastRow}`).fillDown();
  styleHeader(minuteSheet.getRange("A1:N1"));
  styleBody(minuteSheet.getRange(`A2:N${minuteLastRow}`));
  minuteSheet.getRange(`A2:A${minuteLastRow}`).format.numberFormat = "0";
  minuteSheet.getRange(`B2:B${minuteLastRow}`).format.numberFormat = "0.000000";
  minuteSheet.getRange(`C2:C${minuteLastRow}`).format.numberFormat = "0.0000000000";
  minuteSheet.getRange(`D2:E${minuteLastRow}`).format.numberFormat = "0.00000000";
  minuteSheet.getRange(`F2:F${minuteLastRow}`).format.numberFormat = "0.0000000000";
  minuteSheet.getRange(`G2:J${minuteLastRow}`).format.numberFormat = "0.0000000000";
  minuteSheet.getRange(`K2:K${minuteLastRow}`).format.numberFormat = "0.00000000000000";
  minuteSheet.getRange(`M2:M${minuteLastRow}`).format.numberFormat = "0.0000000000";
  minuteSheet.getRange(`N2:N${minuteLastRow}`).format.numberFormat = "0.000000000000";
  minuteSheet.getRange("A:B").format.columnWidth = 14;
  minuteSheet.getRange("C:F").format.columnWidth = 23;
  minuteSheet.getRange("G:J").format.columnWidth = 19;
  minuteSheet.getRange("K:K").format.columnWidth = 22;
  minuteSheet.getRange("L:N").format.columnWidth = 20;
  minuteSheet.freezePanes.freezeRows(1);
  minuteSheet.showGridLines = false;

  sourceSheet.getRange("A1:C1").values = [["时间(s)", "时间(h)", "半径(cm)"]];
  sourceSheet.getRange(`A2:C${attachmentRows.length + 1}`).values = attachmentRows;
  sourceSheet.getRange("E1:F9").values = [
    ["项目", "说明"],
    ["半径预测", "附件2相邻节点分段线性插值"],
    ["公式", "R(t)=R_i+(R_(i+1)-R_i)(t-t_i)/(t_(i+1)-t_i)"],
    ["输出区间", "4 h至72 h，每60 s"],
    ["半径算法改动", "无；与原a4_shrinkage_fvm.py完全一致"],
    ["环境边界", "0--4 h实测；4 h后取3--4 h时间加权均值"],
    ["潜热改动", "只修改移动表面能量平衡，不修改半径插值"],
    ["动态干基密度", "rho_d,s=(760+90*C_s)/(1+C_s)，每次Picard迭代更新"],
    ["温度单位", "附录4扩散系数D中的T使用开尔文(K)"],
  ];
  styleHeader(sourceSheet.getRange("A1:C1"));
  styleHeader(sourceSheet.getRange("E1:F1"));
  styleBody(sourceSheet.getRange(`A2:C${attachmentRows.length + 1}`));
  styleBody(sourceSheet.getRange("E2:F9"));
  sourceSheet.getRange(`A2:A${attachmentRows.length + 1}`).format.numberFormat = "0";
  sourceSheet.getRange(`B2:B${attachmentRows.length + 1}`).format.numberFormat = "0.000000";
  sourceSheet.getRange(`C2:C${attachmentRows.length + 1}`).format.numberFormat = "0.0000000000";
  sourceSheet.getRange("A:C").format.columnWidth = 18;
  sourceSheet.getRange("D:D").format.columnWidth = 3;
  sourceSheet.getRange("E:E").format.columnWidth = 20;
  sourceSheet.getRange("F:F").format.columnWidth = 58;
  sourceSheet.getRange("E2:F9").format.wrapText = true;
  sourceSheet.freezePanes.freezeRows(1);
  sourceSheet.showGridLines = false;

  const initialDryDensity =
    validation.fixed_parameters.initial_dry_bulk_density_kg_m3_for_reference;
  const dynamicDensity = validation.dynamic_surface_density;
  if (
    !(initialDryDensity > 0)
    || !(dynamicDensity.minimum_saved_surface_kg_m3 > 0)
    || !(dynamicDensity.maximum_saved_surface_kg_m3 >= dynamicDensity.minimum_saved_surface_kg_m3)
  ) {
    throw new Error("Validation did not contain a valid dynamic Appendix-4 dry density.");
  }

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return { minuteLastRow, sourceLastRow: attachmentRows.length + 1 };
}

async function readAttachment2(attachmentPath) {
  const workbook = await SpreadsheetFile.importXlsx(
    await FileBlob.load(attachmentPath),
  );
  const sheet = workbook.worksheets.getItemAt(0);
  const used = sheet.getUsedRange(true);
  const values = used.values;
  const rows = values.slice(1).filter((row) => row[0] !== null && row[0] !== "");
  if (rows.length !== 145) {
    throw new Error(`Expected 145 Attachment-2 data nodes, received ${rows.length}.`);
  }
  return rows.map((row) => [Number(row[0]), Number(row[0]) / 3600, Number(row[1])]);
}

async function inspectAndRender(
  result4Path,
  inputsPath,
  previewDir,
  result4LastRow,
  minuteLastRow,
  sourceLastRow,
) {
  await fs.mkdir(previewDir, { recursive: true });
  const result4 = await SpreadsheetFile.importXlsx(await FileBlob.load(result4Path));
  const inputs = await SpreadsheetFile.importXlsx(await FileBlob.load(inputsPath));
  const reports = [];
  for (const [workbook, sheetName, range] of [
    [result4, "Sheet1", "A1:W5"],
    [result4, "Sheet1", `A${result4LastRow - 2}:W${result4LastRow}`],
    [inputs, "逐分钟输入", "A1:N6"],
    [inputs, "逐分钟输入", `A${minuteLastRow - 3}:N${minuteLastRow}`],
    [inputs, "附件2节点", "A1:F10"],
    [inputs, "附件2节点", `A${sourceLastRow - 3}:C${sourceLastRow}`],
  ]) {
    const inspection = await workbook.inspect({
      kind: "table",
      sheetId: sheetName,
      range,
      maxChars: 12000,
      tableMaxRows: 10,
      tableMaxCols: 23,
      include: "values,formulas",
    });
    reports.push(inspection.ndjson);
  }
  for (const [label, workbook] of [["result4", result4], ["inputs", inputs]]) {
    const errors = await workbook.inspect({
      kind: "match",
      searchTerm:
        "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
      options: { useRegex: true, maxResults: 100 },
      summary: `${label} formula error scan`,
    });
    reports.push(errors.ndjson);
  }
  await fs.writeFile(
    path.join(previewDir, "a4_steam_workbooks_inspection.ndjson"),
    `${reports.join("\n")}\n`,
    "utf8",
  );

  for (const [label, workbook, sheetName, range] of [
    ["result4_top", result4, "Sheet1", "A1:W8"],
    ["result4_bottom", result4, "Sheet1", `A${result4LastRow - 6}:W${result4LastRow}`],
    ["inputs_top", inputs, "逐分钟输入", "A1:N8"],
    ["inputs_bottom", inputs, "逐分钟输入", `A${minuteLastRow - 6}:N${minuteLastRow}`],
    ["attachment2_top", inputs, "附件2节点", "A1:F12"],
    ["attachment2_bottom", inputs, "附件2节点", `A${sourceLastRow - 6}:C${sourceLastRow}`],
  ]) {
    const preview = await workbook.render({
      sheetName,
      range,
      scale: 1.25,
      format: "png",
    });
    await fs.writeFile(
      path.join(previewDir, `${label}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}

async function main() {
  const [
    templatePath,
    payloadPath,
    inputsCsvPath,
    attachment2Path,
    validationPath,
    result4Path,
    inputsPath,
    previewDir,
  ] = process.argv.slice(2);
  if (!previewDir) {
    throw new Error(
      "Usage: node a4_steam_build_workbooks.cjs TEMPLATE PAYLOAD INPUT_CSV ATTACHMENT2 VALIDATION RESULT4_XLSX INPUTS_XLSX PREVIEW_DIR",
    );
  }
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const inputRows = parseInputCsv(await fs.readFile(inputsCsvPath, "utf8"));
  const attachmentRows = await readAttachment2(attachment2Path);
  const validation = JSON.parse(await fs.readFile(validationPath, "utf8"));

  const result4Meta = await buildResult4(templatePath, payload, result4Path);
  const inputsMeta = await buildInputsWorkbook(
    inputRows,
    attachmentRows,
    validation,
    inputsPath,
  );
  await inspectAndRender(
    result4Path,
    inputsPath,
    previewDir,
    result4Meta.lastRow,
    inputsMeta.minuteLastRow,
    inputsMeta.sourceLastRow,
  );
  console.log(
    `Verified result4_steam.xlsx (${result4Meta.rowCount} data rows) and `
      + `q4_inputs_4h_72h.xlsx (${inputRows.length} minute rows).`,
  );
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
