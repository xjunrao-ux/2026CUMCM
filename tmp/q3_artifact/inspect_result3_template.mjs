import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "D:/Users/Xenop/Documents/Github/2026CUMCM";
const path = `${root}/比赛题目/CUMCM2026Problems/A题/附件/附件3/result3.xlsx`;
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
console.log((await workbook.inspect({
  kind: "workbook,sheet,table,formula,drawing",
  maxChars: 8000,
  tableMaxRows: 8,
  tableMaxCols: 24,
})).ndjson);

for (const sheet of workbook.worksheets.items) {
  const preview = await workbook.render({
    sheetName: sheet.name,
    range: "A1:F5",
    scale: 2,
    format: "png",
  });
  await fs.writeFile(
    `${sheet.name}_result3_template.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
