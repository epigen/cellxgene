import copy
import logging
import sys
from http import HTTPStatus
import zlib
import json

from flask import make_response, jsonify, current_app, abort, stream_with_context
from urllib.parse import unquote

from server.common.config.client_config import get_client_config
from server.common.constants import Axis, DiffExpMode, JSON_NaN_to_num_warning_msg
from server.common.errors import (
    FilterError,
    JSONEncodingValueError,
    PrepareError,
    DisabledFeatureError,
    ExceedsLimitError,
    DatasetAccessError,
    ColorFormatException,
    AnnotationsError,
    ObsoleteRequest,
    UnsupportedSummaryMethod,
)
from server.common.genesets import summarizeQueryHash
from server.common.fbs.matrix import decode_matrix_fbs


def abort_and_log(code, logmsg, loglevel=logging.DEBUG, include_exc_info=False):
    """
    Log the message, then abort with HTTP code. If include_exc_info is true,
    also include current exception via sys.exc_info().
    """
    if include_exc_info:
        exc_info = sys.exc_info()
    else:
        exc_info = False
    current_app.logger.log(loglevel, logmsg, exc_info=exc_info)
    # Do NOT send log message to HTTP response.
    return abort(code)


def _query_parameter_to_filter(args):
    """
    Convert an annotation value filter, if present in the query args,
    into the standard dict filter format used by internal code.

    Query param filters look like:  <axis>:name=value, where value
    may be one of:
        - a range, min,max, where either may be an open range by using an asterisk, eg, 10,*
        - a value
    Eg,
        ...?tissue=lung&obs:tissue=heart&obs:num_reads=1000,*
    """
    filters = {
        "obs": {},
        "var": {},
    }

    # args has already been url-unquoted once.  We assume double escaping
    # on name and value.
    try:
        for key, value in args.items(multi=True):
            axis, name = key.split(":")
            if axis not in ("obs", "var"):
                raise FilterError("unknown filter axis")
            name = unquote(name)
            current = filters[axis].setdefault(name, {"name": name})

            val_split = value.split(",")
            if len(val_split) == 1:
                if "min" in current or "max" in current:
                    raise FilterError("do not mix range and value filters")
                value = unquote(value)
                values = current.setdefault("values", [])
                values.append(value)

            elif len(val_split) == 2:
                if len(current) > 1:
                    raise FilterError("duplicate range specification")
                min = unquote(val_split[0])
                max = unquote(val_split[1])
                if min != "*":
                    current["min"] = float(min)
                if max != "*":
                    current["max"] = float(max)
                if len(current) < 2:
                    raise FilterError("must specify at least min or max in range filter")

            else:
                raise FilterError("badly formated filter value")

    except ValueError as e:
        raise FilterError(str(e))

    result = {}
    for axis in ("obs", "var"):
        axis_filter = filters[axis]
        if len(axis_filter) > 0:
            result[axis] = {"annotation_value": [val for val in axis_filter.values()]}

    return result


def schema_get_helper(data_adaptor):
    """helper function to gather the schema from the data source and annotations"""
    schema = data_adaptor.get_schema()
    schema = copy.deepcopy(schema)

    # add label obs annotations as needed
    annotations = data_adaptor.dataset_config.user_annotations
    if annotations.user_annotations_enabled():
        label_schema = annotations.get_schema(data_adaptor)
        schema["annotations"]["obs"]["columns"].extend(label_schema)

    return schema


def schema_get(data_adaptor):
    schema = schema_get_helper(data_adaptor)
    return make_response(jsonify({"schema": schema}), HTTPStatus.OK)


def config_get(app_config, data_adaptor):
    config = get_client_config(app_config, data_adaptor)
    return make_response(jsonify(config), HTTPStatus.OK)


def annotations_obs_get(request, data_adaptor):
    fields = request.args.getlist("annotation-name", None)
    num_columns_requested = len(data_adaptor.get_obs_keys()) if len(fields) == 0 else len(fields)
    if data_adaptor.server_config.exceeds_limit("column_request_max", num_columns_requested):
        return abort(HTTPStatus.BAD_REQUEST)
    preferred_mimetype = request.accept_mimetypes.best_match(["application/octet-stream"])
    if preferred_mimetype != "application/octet-stream":
        return abort(HTTPStatus.NOT_ACCEPTABLE)

    try:
        labels = None
        annotations = data_adaptor.dataset_config.user_annotations
        if annotations.user_annotations_enabled():
            labels = annotations.read_labels(data_adaptor)
        fbs = data_adaptor.annotation_to_fbs_matrix(Axis.OBS, fields, labels)
        return make_response(fbs, HTTPStatus.OK, {"Content-Type": "application/octet-stream"})
    except KeyError as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)


def annotations_put_fbs_helper(data_adaptor, fbs):
    """helper function to write annotations from fbs"""
    annotations = data_adaptor.dataset_config.user_annotations
    if not annotations.user_annotations_enabled():
        raise DisabledFeatureError("Writable annotations are not enabled")

    new_label_df = decode_matrix_fbs(fbs)
    if not new_label_df.empty:
        new_label_df = data_adaptor.check_new_labels(new_label_df)
    annotations.write_labels(new_label_df, data_adaptor)


def inflate(data):
    return zlib.decompress(data)


def annotations_obs_put(request, data_adaptor):
    annotations = data_adaptor.dataset_config.user_annotations
    if not annotations.user_annotations_enabled():
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    anno_collection = request.args.get("annotation-collection-name", default=None)
    fbs = inflate(request.get_data())

    if anno_collection is not None:
        if not annotations.is_safe_collection_name(anno_collection):
            return abort(HTTPStatus.BAD_REQUEST, "Bad annotation collection name")
        annotations.set_collection(anno_collection)

    try:
        annotations_put_fbs_helper(data_adaptor, fbs)
        res = json.dumps({"status": "OK"})
        return make_response(res, HTTPStatus.OK, {"Content-Type": "application/json"})
    except (ValueError, DisabledFeatureError, KeyError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)


def annotations_var_get(request, data_adaptor):
    fields = request.args.getlist("annotation-name", None)
    num_columns_requested = len(data_adaptor.get_var_keys()) if len(fields) == 0 else len(fields)
    if data_adaptor.server_config.exceeds_limit("column_request_max", num_columns_requested):
        return abort(HTTPStatus.BAD_REQUEST)
    preferred_mimetype = request.accept_mimetypes.best_match(["application/octet-stream"])
    if preferred_mimetype != "application/octet-stream":
        return abort(HTTPStatus.NOT_ACCEPTABLE)

    try:
        labels = None
        return make_response(
            data_adaptor.annotation_to_fbs_matrix(Axis.VAR, fields, labels),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )
    except KeyError as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)


def data_var_put(request, data_adaptor):
    preferred_mimetype = request.accept_mimetypes.best_match(["application/octet-stream"])
    if preferred_mimetype != "application/octet-stream":
        return abort(HTTPStatus.NOT_ACCEPTABLE)

    filter_json = request.get_json()
    filter = filter_json["filter"] if filter_json else None
    try:
        return make_response(
            data_adaptor.data_frame_to_fbs_matrix(filter, axis=Axis.VAR),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )
    except (FilterError, ValueError, ExceedsLimitError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)


def data_var_get(request, data_adaptor):
    preferred_mimetype = request.accept_mimetypes.best_match(["application/octet-stream"])
    if preferred_mimetype != "application/octet-stream":
        return abort(HTTPStatus.NOT_ACCEPTABLE)

    try:
        filter = _query_parameter_to_filter(request.args)
        return make_response(
            data_adaptor.data_frame_to_fbs_matrix(filter, axis=Axis.VAR),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )
    except (FilterError, ValueError, ExceedsLimitError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)


def colors_get(data_adaptor):
    if not data_adaptor.dataset_config.presentation__custom_colors:
        return make_response(jsonify({}), HTTPStatus.OK)
    try:
        return make_response(jsonify(data_adaptor.get_colors()), HTTPStatus.OK)
    except ColorFormatException as e:
        return abort_and_log(HTTPStatus.NOT_FOUND, str(e), include_exc_info=True)


def diffexp_obs_post(request, data_adaptor):
    if not data_adaptor.dataset_config.diffexp__enable:
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    args = request.get_json()
    try:
        # TODO: implement varfilter mode
        mode = DiffExpMode(args["mode"])

        if mode == DiffExpMode.VAR_FILTER or "varFilter" in args:
            return abort_and_log(HTTPStatus.NOT_IMPLEMENTED, "varFilter not enabled")

        set1_filter = args.get("set1", {"filter": {}})["filter"]
        set2_filter = args.get("set2", {"filter": {}})["filter"]
        count = args.get("count", None)

        if set1_filter is None or set2_filter is None or count is None:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "missing required parameter")
        if Axis.VAR in set1_filter or Axis.VAR in set2_filter:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "var axis filter not enabled")

    except (KeyError, TypeError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)

    try:
        diffexp = data_adaptor.diffexp_topN(set1_filter, set2_filter, count)
        return make_response(diffexp, HTTPStatus.OK, {"Content-Type": "application/json"})
    except (ValueError, DisabledFeatureError, FilterError, ExceedsLimitError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)
    except JSONEncodingValueError:
        # JSON encoding failure, usually due to bad data. Just let it ripple up
        # to default exception handler.
        current_app.logger.warning(JSON_NaN_to_num_warning_msg)
        raise


def layout_obs_get(request, data_adaptor):
    fields = request.args.getlist("layout-name", None)
    num_columns_requested = len(data_adaptor.get_embedding_names()) if len(fields) == 0 else len(fields)
    if data_adaptor.server_config.exceeds_limit("column_request_max", num_columns_requested):
        return abort(HTTPStatus.BAD_REQUEST)

    preferred_mimetype = request.accept_mimetypes.best_match(["application/octet-stream"])
    if preferred_mimetype != "application/octet-stream":
        return abort(HTTPStatus.NOT_ACCEPTABLE)

    try:
        return make_response(
            data_adaptor.layout_to_fbs_matrix(fields), HTTPStatus.OK, {"Content-Type": "application/octet-stream"}
        )
    except (KeyError, DatasetAccessError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)
    except PrepareError:
        return abort_and_log(
            HTTPStatus.NOT_IMPLEMENTED,
            f"No embedding available {request.path}",
            loglevel=logging.ERROR,
            include_exc_info=True,
        )


def genesets_get(request, data_adaptor):
    preferred_mimetype = request.accept_mimetypes.best_match(["application/json", "text/csv"])
    if preferred_mimetype not in ("application/json", "text/csv"):
        return abort(HTTPStatus.NOT_ACCEPTABLE)

    try:
        annotations = data_adaptor.dataset_config.user_annotations
        (genesets, tid) = annotations.read_gene_sets(data_adaptor)

        if preferred_mimetype == "text/csv":
            return make_response(
                annotations.gene_sets_to_csv(genesets),
                HTTPStatus.OK,
                {
                    "Content-Type": "text/csv",
                    "Content-Disposition": "attachment; filename=genesets.csv",
                },
            )
        else:
            return make_response(
                jsonify({"genesets": annotations.gene_sets_to_response(genesets), "tid": tid}), HTTPStatus.OK
            )
    except (ValueError, KeyError, AnnotationsError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e))


def genesets_put(request, data_adaptor):
    annotations = data_adaptor.dataset_config.user_annotations
    if not annotations.gene_sets_save_enabled():
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    anno_collection = request.args.get("annotation-collection-name", default=None)
    if anno_collection is not None:
        if not annotations.is_safe_collection_name(anno_collection):
            return abort(HTTPStatus.BAD_REQUEST, "Bad annotation collection name")
        annotations.set_collection(anno_collection)

    args = request.get_json()
    try:
        genesets = args.get("genesets", None)
        tid = args.get("tid", None)
        if genesets is None:
            abort(HTTPStatus.BAD_REQUEST)

        annotations.write_gene_sets(genesets, tid, data_adaptor)
        return make_response(jsonify({"status": "OK"}), HTTPStatus.OK)
    except (ValueError, DisabledFeatureError, KeyError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)
    except (ObsoleteRequest, TypeError) as e:
        return abort(HTTPStatus.NOT_FOUND, description=str(e))


def summarize_var_helper(request, data_adaptor, key, raw_query):
    preferred_mimetype = request.accept_mimetypes.best_match(["application/octet-stream"])
    if preferred_mimetype != "application/octet-stream":
        return abort(HTTPStatus.NOT_ACCEPTABLE)

    summary_method = request.values.get("method", default="mean")
    query_hash = summarizeQueryHash(raw_query)
    if key and query_hash != key:
        return abort(HTTPStatus.BAD_REQUEST, description="query key did not match")

    args_filter_only = request.values.copy()
    args_filter_only.poplist("method")
    args_filter_only.poplist("key")

    try:
        filter = _query_parameter_to_filter(args_filter_only)
        return make_response(
            data_adaptor.summarize_var(summary_method, filter, query_hash),
            HTTPStatus.OK,
            {"Content-Type": "application/octet-stream"},
        )
    except ValueError as e:
        return abort(HTTPStatus.NOT_FOUND, description=str(e))
    except (UnsupportedSummaryMethod, FilterError) as e:
        return abort(HTTPStatus.BAD_REQUEST, description=str(e))


def summarize_var_get(request, data_adaptor):
    return summarize_var_helper(request, data_adaptor, None, request.query_string)


def summarize_var_post(request, data_adaptor):
    if not request.content_type or "application/x-www-form-urlencoded" not in request.content_type:
        return abort(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
    if request.content_length > 1_000_000:  # just a sanity check to avoid memory exhaustion
        return abort(HTTPStatus.BAD_REQUEST)

    key = request.args.get("key", default=None)
    return summarize_var_helper(request, data_adaptor, key, request.get_data())


def llm_embeddings_text_post(request, data_adaptor):
    """
    Given a text description, return a cell annotation
    """
    try:
        llm_embeddings_text_post.counter
    except AttributeError:
        llm_embeddings_text_post.counter = 0

    if not data_adaptor.dataset_config.llmembs__enable:
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    args = request.get_json()
    try:
        text = args.get("text")

        if text is None:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "missing required parameter text")

    except (KeyError, TypeError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)

    try:
        labels = data_adaptor.compute_llmembs_text_to_annotations(text)

        # compute a string-like hash and take the first 5 characters

        annotation_name = f"{llm_embeddings_text_post.counter}_{text.replace(' ', '_')[:20]}"  # TODO could be used for HTML injection? replace more stuff
        llm_embeddings_text_post.counter += 1

        labels.name = annotation_name
        index_name = data_adaptor.parameters.get("obs_names")
        labels.index = data_adaptor.data.obs[index_name]

        if request.accept_mimetypes.accept_json:
            return make_response(jsonify(labels.to_dict()), HTTPStatus.OK)

        fbs = data_adaptor.annotation_to_fbs_matrix(
            Axis.OBS, [annotation_name], labels.to_frame()
        )  # same as calling encode_matrix_fbs directly

        return make_response(fbs, HTTPStatus.OK, {"Content-Type": "application/octet-stream"})

    except (ValueError, DisabledFeatureError, FilterError, ExceedsLimitError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)
    except JSONEncodingValueError:
        # JSON encoding failure, usually due to bad data. Just let it ripple up
        # to default exception handler.
        current_app.logger.warning(JSON_NaN_to_num_warning_msg)
        raise


def llm_embeddings_obs_post(request, data_adaptor):
    """
    Given a set of cells, return a text description for them
    """
    if not data_adaptor.dataset_config.llmembs__enable:
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    args = request.get_json()
    try:
        selection_filter = args.get("cellSelection", {"filter": {}})["filter"]

        if selection_filter is None:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "missing required parameter set1")
        if Axis.VAR in selection_filter:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "var axis filter not enabled")

    except (KeyError, TypeError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)

    try:
        model_result = data_adaptor.llmembs_obs_to_text(selection_filter)
        return make_response(model_result, HTTPStatus.OK, {"Content-Type": "application/json"})
    except (ValueError, DisabledFeatureError, FilterError, ExceedsLimitError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)
    except JSONEncodingValueError:
        # JSON encoding failure, usually due to bad data. Just let it ripple up
        # to default exception handler.
        current_app.logger.warning(JSON_NaN_to_num_warning_msg)
        raise


def llm_embeddings_chat_post(request, data_adaptor):
    if not data_adaptor.dataset_config.llmembs__enable:
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    args = request.get_json()
    try:
        selection_filter = args.get("cellSelection", {"filter": {}})["filter"]

        # TODO more filters may be appropriate
        if selection_filter is None:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "missing required parameter cellSelection")

    except (KeyError, TypeError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)

    try:
        chat_generator = data_adaptor.establish_llmembs_chat(args, selection_filter)

        response = make_response(
            stream_with_context(chat_generator), HTTPStatus.OK, {"Content-Type": "application/json"}
        )
        response.headers["Content-Encoding"] = "identity"  # Explicitly set to 'identity' to indicate no compression
        return response
    except (ValueError, DisabledFeatureError, FilterError, ExceedsLimitError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)
    except JSONEncodingValueError:
        # JSON encoding failure, usually due to bad data. Just let it ripple up
        # to default exception handler.
        current_app.logger.warning(JSON_NaN_to_num_warning_msg)
        raise


def llm_embeddings_feedback_post(request, data_adaptor):
    if not data_adaptor.dataset_config.llmembs__enable:
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    args = request.get_json()
    try:
        args["thumbDirection"]
        args["messages"]
        selection_filter = args["cellSelection"]["filter"]
    except (KeyError, TypeError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)

    try:
        data_adaptor.llmembs_feedback(args, selection_filter)
        return make_response(jsonify({"status": "OK"}), HTTPStatus.OK)
    except (ValueError, DisabledFeatureError, FilterError, ExceedsLimitError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)
    except JSONEncodingValueError:
        # JSON encoding failure, usually due to bad data. Just let it ripple up
        # to default exception handler.
        current_app.logger.warning(JSON_NaN_to_num_warning_msg)
        raise


def llm_embeddings_gene_contributions_post(request, data_adaptor):
    if not data_adaptor.dataset_config.llmembs__enable:
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    if not data_adaptor.dataset_config.llmembs__gene_score_contribution_enable:
        return abort(HTTPStatus.NOT_IMPLEMENTED)

    args = request.get_json()
    try:
        text = args.get("text")

        if text is None:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "missing required parameter text")

        selection_filter = args.get("cellSelection", {"filter": {}})["filter"]

        if selection_filter is None:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "missing required parameter set1")
        if Axis.VAR in selection_filter:
            return abort_and_log(HTTPStatus.BAD_REQUEST, "var axis filter not enabled")
    except (KeyError, TypeError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)

    try:
        gene_score_contributions = data_adaptor.compute_gene_score_contributions(text, selection_filter)
        return make_response(jsonify(gene_score_contributions.to_dict()), HTTPStatus.OK)
    except (ValueError, DisabledFeatureError, FilterError, ExceedsLimitError) as e:
        return abort_and_log(HTTPStatus.BAD_REQUEST, str(e), include_exc_info=True)
    except JSONEncodingValueError:
        # JSON encoding failure, usually due to bad data. Just let it ripple up
        # to default exception handler.
        current_app.logger.warning(JSON_NaN_to_num_warning_msg)
        raise
