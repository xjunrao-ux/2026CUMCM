import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = "D:/Users/Xenop/Documents/Github/2026CUMCM";
const templatePath = `${root}/比赛题目/CUMCM2026Problems/A题/附件/附件3/result2.xlsx`;
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(templatePath));

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table,formula,drawing",
  maxChars: 10000,
  tableMaxRows: 8,
  tableMaxCols: 24,
});
console.log(inspection.ndjson);

for (const sheetName of ["温度", "水分浓度"]) {
  const preview = await workbook.render({
    sheetName,
    range: "A1:F5",
    scale: 2,
    format: "png",
  });
  await fs.writeFile(
    `${sheetName}_result2_template.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
