import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const [outDir, ...files] = process.argv.slice(2);
await fs.mkdir(outDir, { recursive: true });
for (const file of files) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(file));
  const summary = await workbook.inspect({
    kind: "workbook,sheet,region",
    maxChars: 12000,
    tableMaxRows: 30,
    tableMaxCols: 20,
    tableMaxCellChars: 120,
  });
  console.log(`===== ${path.basename(file)} =====`);
  console.log(summary.ndjson);
  const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
  for (const line of sheets.ndjson.trim().split(/\r?\n/)) {
    if (!line) continue;
    const record = JSON.parse(line);
    const name = record.name;
    if (!name) continue;
    const preview = await workbook.render({ sheetName: name, autoCrop: "all", scale: 2, format: "png" });
    const safe = `${path.basename(file, ".xlsx")}-${name.replace(/[\\/:*?"<>|]/g, "_")}.png`;
    await fs.writeFile(path.join(outDir, safe), new Uint8Array(await preview.arrayBuffer()));
  }
}
