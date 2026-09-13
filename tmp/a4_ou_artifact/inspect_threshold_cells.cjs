"use strict";

const { FileBlob, SpreadsheetFile } = require("@oai/artifact-tool");

async function main() {
  const [workbookPath, range] = process.argv.slice(2);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
  const report = await workbook.inspect({
    kind: "table",
    sheetId: "Sheet1",
    range,
    include: "values,formulas",
    maxChars: 5000,
    tableMaxRows: 10,
    tableMaxCols: 5,
  });
  console.log(report.ndjson);
}

main().catch((error) => {
  console.error(error.stack || String(error));
  process.exitCode = 1;
});
