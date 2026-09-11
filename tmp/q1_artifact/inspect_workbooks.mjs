import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = "D:/Users/Xenop/Documents/Github/2026CUMCM/比赛题目/CUMCM2026Problems/A题/附件/附件1.xlsx";
const templatePath = "D:/Users/Xenop/Documents/Github/2026CUMCM/比赛题目/CUMCM2026Problems/A题/附件/附件3/result1.xlsx";

for (const [label, path] of [["source", sourcePath], ["template", templatePath]]) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
  const summary = await workbook.inspect({
    kind: "workbook,sheet,table,formula,drawing",
    maxChars: 12000,
    tableMaxRows: 12,
    tableMaxCols: 24,
    options: { maxResults: 100 },
  });
  console.log(`---${label}---`);
  console.log(summary.ndjson);

  for (const sheet of workbook.worksheets.items) {
    const used = sheet.getUsedRange();
    if (used) {
      const preview = await workbook.inspect({
        kind: "region",
        sheetId: sheet.name,
        range: used.address.split("!").at(-1),
        maxChars: 12000,
        tableMaxRows: 12,
        tableMaxCols: 24,
      });
      console.log(`---${label}:${sheet.name}---`);
      console.log(preview.ndjson);
    }
  }
}
