/*
Helper functions for the embedded graph colors
*/
import * as d3 from "d3";
import { interpolateRainbow, interpolateCool, interpolateRdBu } from "d3-scale-chromatic";
import memoize from "memoize-one";
import * as globals from "../../globals";
import parseRGB from "../parseRGB";
import { range } from "../range";

/*
given a color mode & accessor, generate an annoMatrix query that will
fulfill it
*/
export function createColorQuery(colorMode, colorByAccessor, schema, genesets) {
  if (!colorMode || !colorByAccessor || !schema || !genesets) return null;

  switch (colorMode) {
  case "color by cellwhisperer search":
  case "color by categorical metadata":
    case "color by continuous metadata": {
      return ["obs", colorByAccessor];
    }
    case "color by expression": {
      const varIndex = schema?.annotations?.var?.index;
      if (!varIndex) return null;
      return [
        "X",
        {
          where: {
            field: "var",
            column: varIndex,
            value: colorByAccessor,
          },
        },
      ];
    }
    case "color by geneset mean expression": {
      const varIndex = schema?.annotations?.var?.index;

      if (!varIndex) return null;
      if (!genesets) return null;

      const _geneset = genesets.get(colorByAccessor);
      const _setGenes = [..._geneset.genes.keys()];

      return [
        "X",
        {
          summarize: {
            method: "mean",
            field: "var",
            column: varIndex,
            values: _setGenes,
          },
        },
      ];
    }
    default: {
      return null;
    }
  }
}

function _defaultColors(nObs) {
  const defaultCellColor = parseRGB(globals.defaultCellColor);
  return {
    rgb: new Array(nObs).fill(defaultCellColor),
    scale: undefined,
  };
}
const defaultColors = memoize(_defaultColors);

/*
create colors scale and RGB array and return as object. Parameters:
  * colorMode - categorical, etc.
  * colorByAccessor - the annotation label name
  * colorByDataframe - the actual color-by data
  * schema - the entire schema
  * userColors - optional user color table
Returns:
  {
    scale: function, mapping label index to color scale
    rgb: cell label to color mapping
  }
*/
function _createColorTable(
  colorMode,
  colorByAccessor,
  colorByData,
  schema,
  userColors = null
) {
  switch (colorMode) {
    case "color by categorical metadata": {
      const data = colorByData.col(colorByAccessor).asArray();
      if (userColors && colorByAccessor in userColors) {
        return createUserColors(data, colorByAccessor, schema, userColors);
      }
      return createColorsByCategoricalMetadata(data, colorByAccessor, schema);
    }
    case "color by continuous metadata": {
      const col = colorByData.col(colorByAccessor);
      const { min, max } = col.summarize();
      return createColorsByContinuousMetadata(col.asArray(), min, max);
    }
    case "color by expression": {
      const col = colorByData.icol(0);
      const { min, max } = col.summarize();
      return createColorsByContinuousMetadata(col.asArray(), min, max);
    }
    case "color by geneset mean expression": {
      const col = colorByData.icol(0);
      const { min, max } = col.summarize();
      return createColorsByContinuousMetadata(col.asArray(), min, max);
    }
    case "color by cellwhisperer search": {
      const col = colorByData.col(colorByAccessor);
      const { min, max } = col.summarize();
      return createColorsByCellwhispererSearch(col.asArray(), min, max);
    }
    default: {
      return defaultColors(schema.dataframe.nObs);
    }
  }
}
export const createColorTable = memoize(_createColorTable);

/**
 * Create two category label-indexed objects:
 *    - colors: maps label to RGB triplet for that label (used by graph, etc)
 *    - scale: function which given label returns d3 color scale for label
 * Order doesn't matter - everything is keyed by label value.
 */
export function loadUserColorConfig(userColors) {
  const convertedUserColors = {};
  Object.keys(userColors).forEach((category) => {
    const [colors, scaleMap] = Object.keys(userColors[category]).reduce(
      (acc, label) => {
        const color = parseRGB(userColors[category][label]);
        acc[0][label] = color;
        acc[1][label] = d3.rgb(255 * color[0], 255 * color[1], 255 * color[2]);
        return acc;
      },
      [{}, {}]
    );
    const scale = (label) => scaleMap[label];
    convertedUserColors[category] = { colors, scale };
  });
  return convertedUserColors;
}

function _createUserColors(data, colorAccessor, schema, userColors) {
  const { colors, scale: scaleByLabel } = userColors[colorAccessor];
  const rgb = createRgbArray(data, colors);

  // color scale function param is INDEX (offset) into schema categories. It is NOT label value.
  // See createColorsByCategoricalMetadata() for another example.
  const { categories } = schema.annotations.obsByName[colorAccessor];
  const categoryMap = new Map();
  categories.forEach((label, idx) => categoryMap.set(idx, label));
  const scale = (idx) => scaleByLabel(categoryMap.get(idx));

  return { rgb, scale };
}
const createUserColors = memoize(_createUserColors);

function _createColorsByCategoricalMetadata(data, colorAccessor, schema) {
  const { categories } = schema.annotations.obsByName[colorAccessor];

  const scale = d3
    .scaleSequential(interpolateRainbow)
    .domain([0, categories.length]);

  /* pre-create colors - much faster than doing it for each obs */
  const colors = categories.reduce((acc, cat, idx) => {
    acc[cat] = parseRGB(scale(idx));
    return acc;
  }, {});

  const rgb = createRgbArray(data, colors);
  return { rgb, scale };
}
const createColorsByCategoricalMetadata = memoize(
  _createColorsByCategoricalMetadata
);

function createRgbArray(data, colors) {
  const rgb = new Array(data.length);
  for (let i = 0, len = data.length; i < len; i += 1) {
    const label = data[i];
    rgb[i] = colors[label];
  }
  return rgb;
}

function _createColorsByContinuousMetadata(data, min, max) {
  const colorBins = 100;
  const scale = d3
    .scaleQuantile()
    .domain([min, max])
    .range(range(colorBins - 1, -1, -1));

  /* pre-create colors - much faster than doing it for each obs */
  const colors = new Array(colorBins);
  for (let i = 0; i < colorBins; i += 1) {
    colors[i] = parseRGB(interpolateCool(i / colorBins));
  }

  const nonFiniteColor = parseRGB(globals.nonFiniteCellColor);
  const rgb = new Array(data.length);
  for (let i = 0, len = data.length; i < len; i += 1) {
    const val = data[i];
    if (Number.isFinite(val)) {
      const c = scale(val);
      rgb[i] = colors[c];
    } else {
      rgb[i] = nonFiniteColor;
    }
  }
  return { rgb, scale, interpolateFn: interpolateCool };
}


function _createColorsByCellwhispererSearch(data, min, max) {
  const colorBins = 100;
  let negativeBins = colorBins;
  let positiveBins = 0;
  let interval, start, end;
  const colors = new Array(colorBins);

  // Determine the number of bins for negative and positive ranges based on their proportion
  if (min < 0 && max > 0) {
    const totalRange = max - min;
    negativeBins = Math.round(((-min) / totalRange) * colorBins);
    positiveBins = colorBins - negativeBins; // Remaining bins for positive values
  } else if (max <= 0) {
    // All values are non-positive, use all bins
    negativeBins = colorBins;
  } else {
    // All values are non-negative, use all bins
    positiveBins = colorBins;
  }

  // by convention the color scale is reversed.
  const scale = d3
        .scaleQuantile()
        .domain([min, max])
        .range(range(colorBins - 1, -1, -1));

  // positiveBins correspond to the range [0.5, 1.0] and negativeBins to the range [0.0, 0.5]. however, the number of bins needs to flip, due to the reverted color scale
  if (positiveBins < negativeBins) {
    interval = 0.5 / negativeBins;
    start = 0.5 - (interval * negativeBins);
    end = 1.0;
  } else {
    interval = 0.5 / positiveBins;
    start = 0.0;
    end = 0.5 + (interval * positiveBins);
  }
  function interpolateRdBuScaled(t) {
    return interpolateRdBu(start + (t * interval * colorBins));
  }

  for (let i = 0; i < colorBins; i += 1) {
    colors[i] = parseRGB(interpolateRdBuScaled(i / (colorBins -1)));
  }

  const nonFiniteColor = parseRGB(globals.nonFiniteCellColor);
  const rgb = new Array(data.length);
  for (let i = 0, len = data.length; i < len; i += 1) {
    const val = data[i];
    if (Number.isFinite(val)) {
      const c = scale(val);
      rgb[i] = colors[c];
    } else {
      rgb[i] = nonFiniteColor;
    }
  }
  return { rgb, scale, interpolateFn: interpolateRdBuScaled };
}


export const createColorsByContinuousMetadata = memoize(
  _createColorsByContinuousMetadata
);

export const createColorsByCellwhispererSearch = memoize(
  _createColorsByCellwhispererSearch
);
