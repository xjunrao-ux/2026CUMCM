import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { FileBlob, SpreadsheetFile } = require("@oai/artifact-tool");

const [templatePath, resultPath, outputDir] = process.argv.slice(2);
await fs.mkdir(outputDir, { recursive: true });

for (const [label, workbookPath] of [
  ["template", templatePath],
  ["current", resultPath],
]) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
  const summary = await workbook.inspect({
    kind: "workbook,sheet,region,computedStyle",
    range: "A1:Z12",
    maxChars: 12000,
    tableMaxRows: 12,
    tableMaxCols: 26,
    tableMaxCellChars: 120,
  });
  await fs.writeFile(path.join(outputDir, `${label}.inspect.ndjson`), summary.ndjson, "utf8");
  for (const sheet of workbook.worksheets.items) {
    const preview = await workbook.render({
      sheetName: sheet.name,
      range: "A1:Z12",
      scale: 1.5,
      format: "png",
    });
    await fs.writeFile(
      path.join(outputDir, `${label}-${sheet.name}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
  }
}
