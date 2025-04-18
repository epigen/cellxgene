import { Colors } from "@blueprintjs/core";
import { dispatchNetworkErrorMessageToUser } from "./util/actionHelpers";
import ENV_DEFAULT from "../../environment.default.json";

/* overflow category values are created  using this string */
export const overflowCategoryLabel = ": all other labels";

/* default "unassigned" value for user-created categorical metadata */
export const unassignedCategoryLabel = "unassigned";

/*
these are default values for configuration the CLI may supply.
See the REST API and CLI specs for more info.
*/
export const configDefaults = {
  features: {},
  displayNames: {},
  parameters: {
    "disable-diffexp": false,
    "diffexp-may-be-slow": false,
  },
  links: {},
};

/*
Most configuration is stored in the reducer.  A handful of values
are global and stored here.  They are typically set by the config
action handler, which pull the information from the backend/CLI.
All should be set here to their default value.
*/
export const globalConfig = {
  /* if a categorical metadata field has more options than this, truncate */
  maxCategoricalOptionsToDisplay: 200,
};

/* colors */
export const blue = Colors.BLUE3;
export const linkBlue = Colors.BLUE5;
export const lightestGrey = "rgb(249,249,249)";
export const lighterGrey = "rgb(245,245,245)";
export const lightGrey = Colors.LIGHT_GRAY1;
export const mediumGrey = "rgb(153,153,153)";
export const darkGrey = "rgb(102,102,102)";
export const darkerGrey = "rgb(51,51,51)";

export const brightBlue = "#4a90e2";
export const brightGreen = "#A2D729";
export const darkGreen = "#448C4D";

export const nonFiniteCellColor = lightGrey;
export const defaultCellColor = "rgb(0,0,0,1)";
export const logoColor = "black"; /* logo pink: "#E9429A" */

/* typography constants */

export const tiniestFontSize = 12;
export const largestFontSize = 24;
export const uppercaseLetterSpacing = "0.04em";
export const bolder = 700;
export const accentFont = "Georgia,Times,Times New Roman,serif";
export const maxParagraphWidth = 600;

/* layout styles constants */

export const cellxgeneTitleLeftPadding = 14;
export const cellxgeneTitleTopPadding = 7;

export const datasetTitleMaxCharacterCount = 25;

export const maxControlsWidth = 800;

export const graphMargin = { top: 20, right: 10, bottom: 30, left: 40 };
export const graphWidth = 700;
export const graphHeight = 700;
export const scatterplotMarginLeft = 11;

export const rightSidebarWidth = 515;
export const leftSidebarWidth = 325;
export const leftSidebarSectionHeading = {
  fontSize: 18,
  textTransform: "uppercase",
  fontWeight: 500,
  letterSpacing: ".05em",
};
export const leftSidebarSectionPadding = 10;
export const categoryLabelDisplayStringLongLength = 70;
export const categoryLabelDisplayStringShortLength = 11;
export const categoryDisplayStringMaxLength = 33;

export const maxUserDefinedGenes = 25;
export const maxGenes = 100;

export const diffexpPopNamePrefix1 = "Pop1 high";
export const diffexpPopNamePrefix2 = "Pop2 high";

/* various timing-related behaviors */
export const tooltipHoverOpenDelay = 1000; /* ms delay before a tooltip displays */
export const tooltipHoverOpenDelayQuick = 500;

const CXG_SERVER_PORT =
  process.env.CXG_SERVER_PORT || ENV_DEFAULT.CXG_SERVER_PORT;

let _API;

if (typeof window !== "undefined" && window.CELLXGENE && window.CELLXGENE.API) {
  _API = window.CELLXGENE.API;
} else {
  if (CXG_SERVER_PORT === undefined) {
    const errorMessage = "Please set the CXG_SERVER_PORT environment variable.";
    dispatchNetworkErrorMessageToUser(errorMessage);
    throw new Error(errorMessage);
  }

  _API = {
    // prefix: "http://api.clustering.czi.technology/api/",
    // prefix: "http://tabulamuris.cxg.czi.technology/api/",
    // prefix: "http://api-staging.clustering.czi.technology/api/",
    // prefix: `http://s0-n11.hpc.meduniwien.ac.at:${CXG_SERVER_PORT}/api/`,
    // prefix: `http://localhost:${CXG_SERVER_PORT}/api/`,
    prefix: `https://cellwhisperer.cemm.at/colonic_epithelium/api/`,
    version: "v0.2/",
  };
}

export const demoStateMap = {
  default: '{}', // Define your default initial state here
  injected: '{"config":{"displayNames":null,"features":null,"parameters":null},"annoMatrix":null,"obsCrossfilter":null,"annotations":{"dataCollectionNameIsReadOnly":true,"dataCollectionName":null,"isEditingCategoryName":false,"isEditingLabelName":false,"categoryBeingEdited":null,"categoryAddingNewLabel":null,"labelEditable":{"category":null,"label":null},"promptForFilename":true},"genesets":{"initialized":false,"genesets":{}},"genesetsUI":{"createGenesetModeActive":false,"isEditingGenesetName":false,"isAddingGenesToGeneset":false},"layoutChoice":{"available":[],"currentDimNames":[]},"continuousSelection":{},"graphSelection":{"tool":"lasso","selection":{"mode":"all"}},"colors":{"colorMode":null,"colorAccessor":null},"controls":{"loading":true,"error":null,"userDefinedGenes":[],"userDefinedGenesLoading":false,"resettingInterface":false,"graphInteractionMode":"select","opacityForDeselectedCells":0.2,"scatterplotXXaccessor":null,"scatterplotYYaccessor":null,"graphRenderCounter":0},"differential":{"loading":null,"error":null,"celllist1":null,"celllist2":null},"centroidLabels":{"showLabels":true},"pointDilation":{"metadataField":"","categoryField":""},"autosave":{"obsAnnotationSaveInProgress":false,"lastSavedAnnoMatrix":null,"genesetSaveInProgress":false,"lastSavedGenesets":null,"error":false},"llmEmbeddings":{"messages":"injected","loading":false,"cellwhispererSearches":[]},"@@undoable/past":[],"@@undoable/future":[],"@@undoable/filterState":{"prevAction":{"type":"@@redux/INITe.0.s.i.n.p"}},"@@undoable/pending":null}'
}



export const API = _API;
