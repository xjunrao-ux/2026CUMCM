"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { FileBlob, SpreadsheetFile } = require("@oai/artifact-tool");

function parseNumericCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const rows = lines.slice(1).map((line) => line.split(",").map(Number));
  if (rows.length < 2 || rows.some((row) => row.length !== 22)) {
    throw new Error("Expected at least two data rows and 22 columns.");
  }
  if (rows.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error("The moisture CSV contains a non-finite value.");
  }
  for (let index = 0; index < rows.length - 1; index += 1) {
    if (rows[index][0] !== 60 * (index + 1)) {
      throw new Error(`Unexpected 60 s time sequence at row ${index + 2}.`);
    }
  }
  if (rows.at(-1)[0] <= rows.at(-2)[0]) {
    throw new Error("The final drying-time row must follow the preceding row.");
  }
  return rows;
}

async function main() {
  const [templatePath, csvPath, outputPath, previewDir] = process.argv.slice(2);
  if (!templatePath || !csvPath || !outputPath || !previewDir) {
    throw new Error(
      "Usage: node build_result3_steam_from_csv.cjs TEMPLATE CSV OUTPUT PREVIEW_DIR",
    );
  }

  const rows = parseNumericCsv(await fs.readFile(csvPath, "utf8"));
  const radii = Array.from(
    { length: 21 },
    (_, index) => Number((index / 10).toFixed(1)),
  );
  const lastWorksheetRow = rows.length + 1;

  const workbook = await SpreadsheetFile.importXlsx(
    await FileBlob.load(templatePath),
  );
  const sheet = workbook.worksheets.getItem("Sheet1");
  sheet.getRange(`A1:V${lastWorksheetRow}`).values = [
    ["时间\\到药材中心的距离", ...radii],
    ...rows,
  ];
  sheet.getRange(`A2:A${lastWorksheetRow}`).format.numberFormat = "0";
  // 临界含水率附近保留8位，让严格小于0.15的终点在表内直接可辨。
  sheet.getRange(`B2:V${lastWorksheetRow}`).format.numberFormat = "0.00000000";
  sheet.getRange("A1:V1").format.font = { bold: true };
  sheet.getRange("A1:V1").format.horizontalAlignment = "center";
  sheet.getRange("A1:V1").format.verticalAlignment = "center";
  sheet.getRange(`A2:V${lastWorksheetRow}`).format.horizontalAlignment = "right";
  sheet.getRange("A:A").format.columnWidth = 24;
  sheet.getRange("B:V").format.columnWidth = 15;
  sheet.getRange("1:1").format.rowHeight = 24;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);

  const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const topInspection = await saved.inspect({
    kind: "table",
    sheetId: "Sheet1",
    range: "A1:V5",
    maxChars: 6000,
    tableMaxRows: 5,
    tableMaxCols: 22,
  });
  const bottomInspection = await saved.inspect({
    kind: "table",
    sheetId: "Sheet1",
    range: `A${lastWorksheetRow - 2}:V${lastWorksheetRow}`,
    maxChars: 6000,
    tableMaxRows: 3,
    tableMaxCols: 22,
  });
  const errors = await saved.inspect({
    kind: "match",
    searchTerm:
      "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 100 },
    summary: "final formula error scan",
  });

  await fs.mkdir(previewDir, { recursive: true });
  await fs.writeFile(
    path.join(previewDir, "result3_steam_inspection.ndjson"),
    `${topInspection.ndjson}\n${bottomInspection.ndjson}\n${errors.ndjson}\n`,
    "utf8",
  );
  for (const [label, range] of [
    ["top", "A1:V8"],
    ["bottom", `A${lastWorksheetRow - 7}:V${lastWorksheetRow}`],
  ]) {
    const preview = await saved.render({
      sheetName: "Sheet1",
      range,
      scale: 1.25,
      format: "png",
    });
    await fs.writeFile(
      path.join(previewDir, `result3_steam_${label}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }

  console.log(`Verified result3_steam.xlsx: ${rows.length} data rows.`);
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
