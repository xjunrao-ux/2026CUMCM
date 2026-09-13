import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = process.cwd();
const resultDir = path.join(repoRoot, "result", "A_q4_radius_comparison");
const previewDir = path.join(repoRoot, "tmp", "a4_radius_comparison", "previews");
const outputPath = path.join(resultDir, "A_q4_radius_comparison_tables.xlsx");
const fontFamily = "Microsoft YaHei";
const dark = "#243447";
const blue = "#4C78A8";
const orange = "#E07A5F";
const green = "#2A9D8F";
const light = "#E8EEF3";

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const source = text.replace(/^\uFEFF/, "");
  for (let i = 0; i < source.length; i += 1) {
    const ch = source[i];
    if (quoted) {
      if (ch === '"' && source[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function typed(cell) {
  if (cell === "") return null;
  if (cell === "True") return true;
  if (cell === "False") return false;
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(cell)) return Number(cell);
  return cell;
}

function readTypedCsv(filePath) {
  return fs.readFile(filePath, "utf8").then((text) => parseCsv(text).map((row) => row.map(typed)));
}

function columnName(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function styleTitle(sheet, range, title, subtitle) {
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(range).format.font = { name: fontFamily, size: 15, bold: true, color: dark };
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange("A2").format.font = { name: fontFamily, size: 9, italic: true, color: "#5F6B76" };
  sheet.getRange("A3:I3").format.borders = { bottom: { style: "thin", color: light } };
}

function styleHeader(range) {
  range.format = {
    fill: dark,
    font: { name: fontFamily, size: 9, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
}

function styleBody(range) {
  range.format.font = { name: fontFamily, size: 9, color: "#20252B" };
  range.format.verticalAlignment = "center";
}

const [summaryJsonText, tableRows, fullRows] = await Promise.all([
  fs.readFile(path.join(resultDir, "A_q4_radius_comparison_summary.json"), "utf8"),
  readTypedCsv(path.join(resultDir, "A_q4_radius_comparison_table_6h.csv")),
  readTypedCsv(path.join(resultDir, "A_q4_radius_comparison_full_60s.csv")),
]);
const summary = JSON.parse(summaryJsonText);
const wb = Workbook.create();

const summarySheet = wb.worksheets.add("结论汇总");
summarySheet.showGridLines = false;
styleTitle(
  summarySheet,
  "A1:I1",
  "问题四参数下半径收缩影响对比",
  "两组计算仅改变半径函数：R(t)=2 cm 与附件 2 实测 R(t)"
);
summarySheet.getRange("A5:I7").values = [
  ["情景", "半径规则", "烘干时间/s", "烘干时间/h", "烘干时间/d", "结束半径/cm", "相对恒定半径变化/h", "相对变化", "速度倍率"],
  ["恒定半径", "R(t)=2 cm", summary.constant_radius.drying_time_s, summary.constant_radius.drying_time_h, summary.constant_radius.drying_time_d, 2, 0, 0, 1],
  ["实测收缩半径", "附件 2 R(t)", summary.shrinking_radius.drying_time_s, summary.shrinking_radius.drying_time_h, summary.shrinking_radius.drying_time_d, summary.shrinking_radius.radius_at_endpoint_cm, -summary.shrinkage_impact.time_saved_h, -summary.shrinkage_impact.time_reduction_pct / 100, summary.shrinkage_impact.drying_speed_factor],
];
styleHeader(summarySheet.getRange("A5:I5"));
styleBody(summarySheet.getRange("A6:I7"));
summarySheet.getRange("A6:I6").format.fill = "#EAF2F8";
summarySheet.getRange("A7:I7").format.fill = "#FCEDE8";
summarySheet.getRange("C6:C7").format.numberFormat = "0";
summarySheet.getRange("D6:G7").format.numberFormat = "0.0000";
summarySheet.getRange("H6:H7").format.numberFormat = "0.0%";
summarySheet.getRange("I6:I7").format.numberFormat = "0.0000";

summarySheet.getRange("A10:B15").values = [
  ["影响指标", "计算结果"],
  ["节省时间/h", null],
  ["烘干时间缩短比例", null],
  ["干燥速度倍率", null],
  ["收缩/恒定时间比", null],
  ["收缩结束时半径减小比例", null],
];
summarySheet.getRange("B11:B15").formulas = [
  ["=D6-D7"],
  ["=B11/D6"],
  ["=D6/D7"],
  ["=D7/D6"],
  ["=1-F7/F6"],
];
styleHeader(summarySheet.getRange("A10:B10"));
styleBody(summarySheet.getRange("A11:B15"));
summarySheet.getRange("B11").format.numberFormat = "0.0000";
summarySheet.getRange("B12:B12").format.numberFormat = "0.0%";
summarySheet.getRange("B13:B13").format.numberFormat = "0.0000";
summarySheet.getRange("B14:B15").format.numberFormat = "0.0%";
summarySheet.getRange("B11:B15").format.font = { name: fontFamily, size: 10, bold: true, color: green };

summarySheet.getRange("A18:I20").values = [
  ["解释", "数值"],
  ["收缩情景结束时，恒定半径中心水分浓度/(kg/kg)", summary.shrinkage_impact.fixed_center_at_shrinking_endpoint_kg_kg],
  ["恒定半径中心水分浓度高于阈值/(kg/kg)", summary.shrinkage_impact.fixed_center_excess_over_threshold_at_shrinking_endpoint_kg_kg],
];
styleHeader(summarySheet.getRange("A18:B18"));
styleBody(summarySheet.getRange("A19:B20"));
summarySheet.getRange("B19:B20").format.numberFormat = "0.0000";
summarySheet.getRange("A1:I20").format.rowHeight = 20;
summarySheet.getRange("A1:I20").format.autofitColumns();
summarySheet.getRange("A1:A20").format.columnWidth = 28;
summarySheet.getRange("B1:B20").format.columnWidth = 22;

const chart = summarySheet.charts.add("bar", [summarySheet.getRange("A5:A7"), summarySheet.getRange("D5:D7")]);
chart.title = "烘干时间对比";
chart.titleTextStyle.typeface = fontFamily;
chart.titleTextStyle.fontSize = 13;
chart.hasLegend = false;
chart.xAxis = { axisType: "textAxis", textStyle: { typeface: fontFamily, fontSize: 10 } };
chart.yAxis = { numberFormatCode: "0.0", numberFormatSourceLinked: false, textStyle: { typeface: fontFamily, fontSize: 10 } };
chart.yAxis.title.text = "烘干时间/h";
chart.series.items[0].fill = blue;
chart.setPosition("K4", "Q17");

const tableSheet = wb.worksheets.add("6h对照表");
tableSheet.showGridLines = false;
styleTitle(tableSheet, "A1:K1", "6 h 间隔水分浓度对照表", "每个情景最后一行是其首次满足全域 C<0.15 kg/kg 的时刻；空值表示位置已在收缩表面之外");
const tableHeaders = ["情景", "时间/s", "时间/h", "结束行", "当前半径/cm", "C(0 cm)", "C(0.5 cm)", "C(1.0 cm)", "C(1.5 cm)", "C(2.0 cm)", "C(表面)"];
const scenarioMap = { constant_radius: "恒定半径", shrinking_radius: "实测收缩半径" };
const tableData = tableRows.slice(1).map((row) => [scenarioMap[row[0]], ...row.slice(1)]);
tableSheet.getRangeByIndexes(4, 0, 1, tableHeaders.length).values = [tableHeaders];
tableSheet.getRangeByIndexes(5, 0, tableData.length, tableHeaders.length).values = tableData;
styleHeader(tableSheet.getRangeByIndexes(4, 0, 1, tableHeaders.length));
styleBody(tableSheet.getRangeByIndexes(5, 0, tableData.length, tableHeaders.length));
tableSheet.getRangeByIndexes(5, 1, tableData.length, 1).format.numberFormat = "0";
tableSheet.getRangeByIndexes(5, 2, tableData.length, 1).format.numberFormat = "0.0000";
tableSheet.getRangeByIndexes(5, 4, tableData.length, 7).format.numberFormat = "0.0000";
tableSheet.getRange(`A6:K${5 + tableData.length}`).conditionalFormats.addCustom("=$D6=TRUE", { fill: "#E4F4EF", font: { bold: true, color: "#176B57" } });
tableSheet.freezePanes.freezeRows(5);
tableSheet.freezePanes.freezeColumns(1);
tableSheet.getRange(`A1:K${5 + tableData.length}`).format.autofitColumns();
tableSheet.getRange("A:A").format.columnWidth = 18;
tableSheet.getRange("D:D").format.columnWidth = 10;

const fullSheet = wb.worksheets.add("60s完整对照");
fullSheet.showGridLines = false;
const fullHeaders = fullRows[0];
const fullData = fullRows.slice(1);
fullSheet.getRangeByIndexes(0, 0, 1, fullHeaders.length).values = [fullHeaders];
fullSheet.getRangeByIndexes(1, 0, fullData.length, fullHeaders.length).values = fullData;
styleHeader(fullSheet.getRangeByIndexes(0, 0, 1, fullHeaders.length));
styleBody(fullSheet.getRangeByIndexes(1, 0, fullData.length, fullHeaders.length));
fullSheet.getRangeByIndexes(1, 0, fullData.length, 1).format.numberFormat = "0";
fullSheet.getRangeByIndexes(1, 1, fullData.length, fullHeaders.length - 1).format.numberFormat = "0.000000";
fullSheet.freezePanes.freezeRows(1);
fullSheet.freezePanes.freezeColumns(2);
fullSheet.getRangeByIndexes(0, 0, fullData.length + 1, fullHeaders.length).format.columnWidth = 15;
fullSheet.getRange("A:B").format.columnWidth = 12;

const validationSheet = wb.worksheets.add("数值检验");
validationSheet.showGridLines = false;
styleTitle(validationSheet, "A1:E1", "数值检验", "粗细网格差异远小于半径收缩造成的 78.77 h 时长变化");
validationSheet.getRange("A5:E11").values = [
  ["情景", "检验量", "正式值", "参照值", "差异"],
  ["恒定半径", "烘干时间/s", summary.constant_radius.drying_time_s, summary.constant_radius.drying_time_s + summary.constant_radius.coarse_fine_difference_s, summary.constant_radius.coarse_fine_difference_s],
  ["实测收缩半径", "烘干时间/s", summary.shrinking_radius.drying_time_s, summary.shrinking_radius.drying_time_s + summary.shrinking_radius.coarse_fine_difference_s, summary.shrinking_radius.coarse_fine_difference_s],
  ["恒定半径", "阈值前最大水分", summary.constant_radius.previous_maximum_moisture_kg_kg, 0.15, summary.constant_radius.previous_maximum_moisture_kg_kg - 0.15],
  ["恒定半径", "阈值后最大水分", summary.constant_radius.final_maximum_moisture_kg_kg, 0.15, summary.constant_radius.final_maximum_moisture_kg_kg - 0.15],
  ["实测收缩半径", "阈值前最大水分", summary.shrinking_radius.previous_maximum_moisture_kg_kg, 0.15, summary.shrinking_radius.previous_maximum_moisture_kg_kg - 0.15],
  ["实测收缩半径", "阈值后最大水分", summary.shrinking_radius.final_maximum_moisture_kg_kg, 0.15, summary.shrinking_radius.final_maximum_moisture_kg_kg - 0.15],
];
styleHeader(validationSheet.getRange("A5:E5"));
styleBody(validationSheet.getRange("A6:E11"));
validationSheet.getRange("C6:E7").format.numberFormat = "0";
validationSheet.getRange("C8:E11").format.numberFormat = "0.0000000000";
validationSheet.getRange("A1:E11").format.autofitColumns();
validationSheet.getRange("B:B").format.columnWidth = 22;

const notesSheet = wb.worksheets.add("口径说明");
notesSheet.showGridLines = false;
styleTitle(notesSheet, "A1:B1", "计算口径与字段说明", "本工作簿汇总正式模型输出，不修改赛题附件和既有问题四结果");
notesSheet.getRange("A5:B12").values = [
  ["项目", "说明"],
  ["共同参数", "附录 4 物性、附件 1 环境边界、初始温度 28 °C、初始水分 2.55 kg/kg、h=25 W/(m²·K)、hm=8×10⁻⁷ m/s"],
  ["唯一差异", "恒定尺寸取 R(t)=2 cm；收缩尺寸取附件 2 的分段线性 R(t)"],
  ["终止条件", "所有径向位置水分浓度严格小于 0.15 kg/kg"],
  ["60s完整对照", "fixed_ 和 shrink_ 分别表示恒定半径与收缩半径；delta_fixed_minus_shrink_ 为二者水分浓度差"],
  ["空值规则", "收缩情景中固定物理半径大于当前药材半径时保持为空，不做外推"],
  ["比较解释", "烘干时间差可归因于半径函数变化；不得与问题三附录 3 参数结果直接混为同一基线"],
  ["源数据", "赛题附件 1、附件 2；正式问题四收缩结果；同一求解器生成的恒定半径对照结果"],
];
styleHeader(notesSheet.getRange("A5:B5"));
styleBody(notesSheet.getRange("A6:B12"));
notesSheet.getRange("A6:B12").format.wrapText = true;
notesSheet.getRange("A:A").format.columnWidth = 22;
notesSheet.getRange("B:B").format.columnWidth = 90;
notesSheet.getRange("A6:B12").format.rowHeight = 36;

await fs.mkdir(previewDir, { recursive: true });
const inspectSummary = await wb.inspect({ kind: "table", range: "结论汇总!A1:I20", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 12 });
const inspectTable = await wb.inspect({ kind: "table", range: "6h对照表!A1:K20", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 12 });
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(inspectSummary.ndjson);
console.log(inspectTable.ndjson);
console.log(errors.ndjson);

for (const [sheetName, range] of [
  ["结论汇总", "A1:Q20"],
  ["6h对照表", "A1:K35"],
  ["60s完整对照", "A1:P18"],
  ["数值检验", "A1:E11"],
  ["口径说明", "A1:B12"],
]) {
  const preview = await wb.render({ sheetName, range, scale: 1.5, format: "png" });
  const safeName = sheetName.replace(/[^\p{L}\p{N}_-]+/gu, "_");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
const reopened = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const reopenedSummary = await reopened.inspect({
  kind: "table",
  range: "结论汇总!A5:I15",
  include: "values,formulas",
  tableMaxRows: 15,
  tableMaxCols: 12,
});
const reopenedErrors = await reopened.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "reopened workbook formula error scan",
});
console.log(reopenedSummary.ndjson);
console.log(reopenedErrors.ndjson);
console.log(`Workbook: ${outputPath}`);
