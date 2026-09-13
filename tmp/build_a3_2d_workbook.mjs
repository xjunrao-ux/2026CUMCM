import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = "D:/Users/Xenop/Documents/Github/2026CUMCM";
const dataDir = path.join(repoRoot, "result", "A_problem3_2d_exposed");
const outputPath = path.join(dataDir, "result3_2d_exposed.xlsx");
const previewDir = path.join(repoRoot, "tmp", "a3_2d_workbook_preview");
const fontFamily = "Arial";

function parseCsv(text) {
  return text.replace(/^\uFEFF/, "").trimEnd().split(/\r?\n/).map((line) =>
    line.split(",").map((value) => {
      const trimmed = value.trim();
      if (trimmed !== "" && Number.isFinite(Number(trimmed))) return Number(trimmed);
      return trimmed;
    })
  );
}

async function loadCsv(name) {
  return parseCsv(await fs.readFile(path.join(dataDir, name), "utf8"));
}

function styleFlatTable(sheet, rangeAddress, headerAddress, numberRange, numberFormat) {
  sheet.showGridLines = false;
  const used = sheet.getRange(rangeAddress);
  used.format.font = { name: fontFamily, size: 10, color: "#1F2937" };
  used.format.verticalAlignment = "center";
  const header = sheet.getRange(headerAddress);
  header.format = {
    fill: "#1F4E78",
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
  };
  header.format.rowHeight = 24;
  if (numberRange) sheet.getRange(numberRange).format.numberFormat = numberFormat;
  used.format.autofitColumns();
  used.format.autofitRows();
}

const summary = JSON.parse(await fs.readFile(path.join(dataDir, "summary.json"), "utf8"));
const table5 = await loadCsv("table5_midplane_moisture_bias_corrected.csv");
const midplane = await loadCsv("midplane_moisture_60s_0p1cm.csv");
const axial = await loadCsv("axis_axial_moisture_6h.csv");
const comparison = await loadCsv("comparison_1d_2d_midplane.csv");

const workbook = Workbook.create();

const summarySheet = workbook.worksheets.add("结果摘要");
summarySheet.showGridLines = false;
summarySheet.getRange("A1").values = [["A题第三问：二维端面暴露模型结果"]];
summarySheet.getRange("A1").format = {
  font: { name: fontFamily, size: 16, bold: true, color: "#1F2937" },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
summarySheet.getRange("A1:F1").format.rowHeight = 28;
summarySheet.getRange("A2").values = [["有限圆柱轴对称二维模型；两个圆形端面与侧面采用相同换热、传质系数"]];
summarySheet.getRange("A2").format = {
  font: { name: fontFamily, size: 10, italic: true, color: "#4B5563" },
};
summarySheet.getRange("A2:F2").format.rowHeight = 20;
summarySheet.getRange("A3:F3").format.borders = {
  bottom: { style: "thin", color: "#9CA3AF" },
};

const p = summary.production;
const c = summary.comparison_with_1d;
const b = summary.half_domain_moisture_balance;
const v = summary.coarse_validation;
const metrics = [
  ["指标", "数值", "单位", "说明"],
  ["推荐烘干时间", c.bias_corrected_2d_drying_time_h, "h", "高精度一维结果叠加同网格二维几何增量"],
  ["直接二维网格结果", p.drying_time_h, "h", `${p.nr}×${p.nz_half_length} 半长网格`],
  ["同网格一维基线", c.matched_1d_drying_time_h, "h", "与二维使用相同径向网格和时间推进"],
  ["端面引起的时间缩短", c.matched_time_reduction_s, "s", "同网格二维与一维之差"],
  ["端面引起的相对缩短", c.matched_time_reduction_percent / 100, "", "几何效应"],
  ["端面累计排湿占比", b.end_outflow_fraction, "", "完整圆柱与半域比例相同"],
  ["侧面累计排湿占比", b.side_outflow_fraction, "", "完整圆柱与半域比例相同"],
  ["二维最大水分控制位置", 0, "cm", "中截面轴心 r=0, z=0"],
  ["最终最大水分", p.final_maximum_moisture, "kg/kg", "严格小于0.15"],
  ["水分守恒相对残差", b.relative_residual, "-", "侧面与端面通量均计入"],
  ["粗细二维网格时间差", v.coarse_minus_production_s, "s", "40×20 与 80×40"],
  ["几何效应粗细网格差", v.geometry_reduction_difference_vs_production_s, "s", "端面增量的网格稳定性"],
];
summarySheet.getRange(`A5:D${4 + metrics.length}`).values = metrics;
styleFlatTable(summarySheet, `A5:D${4 + metrics.length}`, "A5:D5", `B6:B${4 + metrics.length}`, "0.0000");
summarySheet.getRange("B6:B8").format.numberFormat = "0.0000";
summarySheet.getRange("B9").format.numberFormat = "0";
summarySheet.getRange("B10:B12").format.numberFormat = "0.0000%";
summarySheet.getRange("B13").format.numberFormat = "0.0";
summarySheet.getRange("B14").format.numberFormat = "0.0000000000";
summarySheet.getRange("B15").format.numberFormat = "0.000E+00";
summarySheet.getRange("B16:B17").format.numberFormat = "0";
summarySheet.getRange("A5:D5").format.borders = { preset: "inside", style: "thin", color: "#FFFFFF" };
summarySheet.getRange("A19:D19").values = [["模型设定", "取值", "单位", "备注"]];
summarySheet.getRange("A20:D25").values = [
  ["圆柱半径", summary.model_scope.radius_m, "m", "固定半径"],
  ["圆柱长度", summary.model_scope.length_m, "m", "计算半长并利用对称性"],
  ["换热系数", summary.model_scope.side_heat_transfer_w_m2_k, "W/(m²·K)", "侧面与端面相同"],
  ["传质系数", summary.model_scope.side_mass_transfer_m_s, "m/s", "侧面与端面相同"],
  ["干燥阈值", summary.model_scope.drying_threshold_kg_kg, "kg/kg", "二维全域严格低于阈值"],
  ["4 h后环境水分", summary.constant_boundary.moisture_kg_kg, "kg/kg", "附件1最后1 h时间平均"],
];
styleFlatTable(summarySheet, "A19:D25", "A19:D19", "B20:B25", "0.00000000");
summarySheet.getRange("B23").format.numberFormat = "0.000E+00";
summarySheet.getRange("A27:D30").values = [
  ["结论", "内容", "", ""],
  ["最终时间", "建议采用57.5025 h；端面使高精度一维结果缩短约8 s。", "", ""],
  ["空间影响", "端部区域显著更干，但中截面与一维结果最大仅相差约10⁻⁵ kg/kg。", "", ""],
  ["控制位置", "最湿位置由一维轴线收敛为中截面轴心点，最终时间仍由该点控制。", "", ""],
];
summarySheet.getRange("A27:D27").format = {
  fill: "#D9EAF7",
  font: { name: fontFamily, size: 10, bold: true, color: "#1F2937" },
};
summarySheet.getRange("B28:B30").format.wrapText = true;
summarySheet.getRange("A:A").format.columnWidth = 25;
summarySheet.getRange("B:B").format.columnWidth = 38;
summarySheet.getRange("C:C").format.columnWidth = 16;
summarySheet.getRange("D:D").format.columnWidth = 48;

const tableSheet = workbook.worksheets.add("表5校正结果");
tableSheet.getRangeByIndexes(0, 0, table5.length, table5[0].length).values = table5;
styleFlatTable(tableSheet, `A1:F${table5.length}`, "A1:F1", `B2:F${table5.length}`, "0.0000");
tableSheet.freezePanes.freezeRows(1);
tableSheet.getRange("A:A").format.columnWidth = 18;
tableSheet.getRange("B:F").format.columnWidth = 13;
tableSheet.getRange(`A${table5.length}:F${table5.length}`).format.font = {
  name: fontFamily,
  size: 10,
  bold: true,
  color: "#1F2937",
};

const midSheet = workbook.worksheets.add("中截面60s");
midSheet.getRangeByIndexes(0, 0, midplane.length, midplane[0].length).values = midplane;
styleFlatTable(midSheet, `A1:V${midplane.length}`, "A1:V1", `A2:V${midplane.length}`, "0.0000000000");
midSheet.getRange(`A2:A${midplane.length}`).format.numberFormat = "0";
midSheet.freezePanes.freezeRows(1);
midSheet.freezePanes.freezeColumns(1);

const axialSheet = workbook.worksheets.add("轴线6h");
axialSheet.getRangeByIndexes(0, 0, axial.length, axial[0].length).values = axial;
styleFlatTable(axialSheet, `A1:AA${axial.length}`, "A1:AA1", `A2:AA${axial.length}`, "0.0000000000");
axialSheet.getRange(`A2:A${axial.length}`).format.numberFormat = "0.0000";
axialSheet.freezePanes.freezeRows(1);
axialSheet.freezePanes.freezeColumns(1);

const compareSheet = workbook.worksheets.add("一维二维对照");
compareSheet.getRangeByIndexes(0, 0, comparison.length, comparison[0].length).values = comparison;
styleFlatTable(compareSheet, `A1:E${comparison.length}`, "A1:E1", `A2:E${comparison.length}`, "0.0000000000");
compareSheet.getRange(`A2:B${comparison.length}`).format.numberFormat = "0.0";
compareSheet.getRange("A:A").format.columnWidth = 12;
compareSheet.getRange("B:B").format.columnWidth = 14;
compareSheet.getRange("C:C").format.columnWidth = 23;
compareSheet.getRange("D:D").format.columnWidth = 30;
compareSheet.getRange("E:E").format.columnWidth = 27;
compareSheet.freezePanes.freezeRows(1);

await fs.mkdir(previewDir, { recursive: true });
const previewSpecs = [
  ["结果摘要", "A1:D30", "summary.png"],
  ["表5校正结果", `A1:F${table5.length}`, "table5.png"],
  ["中截面60s", "A1:V22", "midplane.png"],
  ["轴线6h", `A1:AA${axial.length}`, "axial.png"],
  ["一维二维对照", "A1:E25", "comparison.png"],
];
for (const [sheetName, range, fileName] of previewSpecs) {
  const preview = await workbook.render({ sheetName, range, scale: 1.25, format: "png" });
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}

const keyInspection = await workbook.inspect({
  kind: "table",
  range: "结果摘要!A1:D30",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 4,
});
await fs.writeFile(path.join(previewDir, "summary_inspect.ndjson"), keyInspection.ndjson, "utf8");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(previewDir, "error_scan.ndjson"), errors.ndjson, "utf8");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, previewDir, sheetCount: 5 }));
