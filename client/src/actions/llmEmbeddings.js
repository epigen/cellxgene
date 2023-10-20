import * as globals from "../globals";
import { annotationCreateContinuousAction } from "./annotation";
import { matrixFBSToDataframe } from "../util/stateManager/matrix";

/*
  LLM embedding querying
*/
export const requestEmbeddingLLMWithCells =
  /*
    Send a request to the LLM embedding model with text
  */
  (set1) => async (dispatch) => {
    dispatch({
      type: "request text to embedding model started",
    });
    try {
      // Legal values are null, Array or TypedArray.  Null is initial state.
      if (!set1) set1 = []; // TODO raise an exception, as we need a selection

      // These lines ensure that we convert any TypedArray to an Array.
      // This is necessary because JSON.stringify() does some very strange
      // things with TypedArrays (they are marshalled to JSON objects, rather
      // than being marshalled as a JSON array).
      set1 = Array.isArray(set1) ? set1 : Array.from(set1);

      const res = await fetch(
        `${globals.API.prefix}${globals.API.version}llmembs/obs`,
        {
          method: "POST",
          headers: new Headers({
            Accept: "application/json",
            "Content-Type": "application/json",
          }),
          body: JSON.stringify({
            set1: { filter: { obs: { index: set1 } } },
          }),
          credentials: "include",
        }
      );

      if (!res.ok || res.headers.get("Content-Type") !== "application/json") {
        return dispatch({
          type: "request llm embeddings error",
          error: new Error(
            `Unexpected response ${res.status} ${
              res.statusText
            } ${res.headers.get("Content-Type")}}`
          ),
        });
      }

      const response = await res.json();
      return dispatch({
        type: "embedding model text response from cells",
        data: response.text,
      });
    } catch (error) {
      return dispatch({
        type: "request llm embeddings error",
        error,
      });
    }
  };

export const requestEmbeddingLLMWithText =
  /*
    Send a request to the LLM embedding model with text
  */
  (text) => async (dispatch) => {
    dispatch({
      type: "request to embedding model started",
    });
    try {
      const res = await fetch(
        `${globals.API.prefix}${globals.API.version}llmembs/text`,
        {
          method: "POST",
          headers: new Headers({
            Accept: "application/octet-stream",
            "Content-Type": "application/json",
          }),
          body: JSON.stringify({
            text,
          }),
          credentials: "include",
        }
      );

      if (
        !res.ok ||
        res.headers.get("Content-Type") !== "application/octet-stream"
      ) {
        return dispatch({
          type: "request llm embeddings error",
          error: new Error(
            `Unexpected response ${res.status} ${
              res.statusText
            } ${res.headers.get("Content-Type")}}`
          ),
        });
      }

      // TODO process the annotation here (e.g. convert to dataframe etc)

      const buffer = await res.arrayBuffer();
      const dataframe = matrixFBSToDataframe(buffer);
      const col = dataframe.__columns[0];

      const annotationName = dataframe.colIndex.getLabel(0);

      dispatch({
        type: "embedding model annotation response from text",
      });

      return dispatch(annotationCreateContinuousAction(annotationName, col));
    } catch (error) {
      return dispatch({
        type: "request llm embeddings error",
        error,
      });
    }
  };
