import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "D:/Users/Xenop/Documents/Github/2026CUMCM";
const templatePath = `${root}/比赛题目/CUMCM2026Problems/A题/附件/附件3/result2.xlsx`;
const resultDir = `${root}/results/A_problem2_coupled`;
const outputPath = `${resultDir}/result2.xlsx`;

function parseNumericCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const rows = lines.slice(1).map((line) => line.split(",").map(Number));
  if (rows.length !== 10801 || rows.some((row) => row.length !== 22)) {
    throw new Error("Expected 10801 data rows and 22 columns in each result CSV.");
  }
  return rows;
}

const temperatureRows = parseNumericCsv(
  await fs.readFile(`${resultDir}/temperature_full_1s_0p1cm.csv`, "utf8"),
);
const moistureRows = parseNumericCsv(
  await fs.readFile(`${resultDir}/moisture_full_1s_0p1cm.csv`, "utf8"),
);
const radii = Array.from({ length: 21 }, (_, index) => Number((index / 10).toFixed(1)));

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));
for (const [sheetName, rows] of [
  ["温度", temperatureRows],
  ["水分浓度", moistureRows],
]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const matrix = [["时间\\到药材中心的距离", ...radii], ...rows];
  sheet.getRange("A1:V10802").values = matrix;
  sheet.getRange("A2:A10802").format.numberFormat = "0";
  sheet.getRange("B2:V10802").format.numberFormat = "0.0000";
  sheet.getRange("A1:V1").format.font = { bold: true };
  sheet.getRange("A1:V1").format.horizontalAlignment = "center";
  sheet.getRange("A1:V1").format.verticalAlignment = "center";
  sheet.getRange("A2:V10802").format.horizontalAlignment = "right";
  sheet.getRange("A:A").format.columnWidth = 24;
  sheet.getRange("B:V").format.columnWidth = 11;
  sheet.getRange("1:1").format.rowHeight = 24;
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(1);
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const inspection = await saved.inspect({
  kind: "sheet,table,formula",
  maxChars: 5000,
  tableMaxRows: 5,
  tableMaxCols: 22,
});
console.log(inspection.ndjson);

for (const [sheetName, row] of [
  ["温度", 10802],
  ["水分浓度", 10802],
]) {
  const selected = await saved.inspect({
    kind: "table",
    sheetId: sheetName,
    range: `A${row}:V${row}`,
    maxChars: 3000,
    tableMaxRows: 2,
    tableMaxCols: 22,
  });
  console.log(selected.ndjson);
}

const errors = await saved.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

for (const sheetName of ["温度", "水分浓度"]) {
  for (const [label, range] of [["top", "A1:V8"], ["bottom", "A10795:V10802"]]) {
    const preview = await saved.render({ sheetName, range, scale: 1.25, format: "png" });
    await fs.writeFile(
      `${sheetName}_${label}_check.png`,
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}

console.log(outputPath);
