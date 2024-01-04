import React from "react";
import { connect } from "react-redux";
import { Button, InputGroup } from "@blueprintjs/core";
import actions from "../../actions";
import * as globals from "../../globals";

function renderList(items) {
  return (
    <ul>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

@connect((state) => ({
  ...state.llmEmbeddings,
  obsCrossfilter: state.obsCrossfilter,
}))
class BottomSideBar extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      inputText: "",
    };
  }

  handleInputChange = (e) => {
    this.setState({ inputText: e.target.value });
  };

  findCellsClick = () => {
    const { dispatch } = this.props;
    const { inputText } = this.state;
    dispatch(actions.requestEmbeddingLLMWithText(inputText));
  };

  describeSelectedClick = () => {
    const { dispatch, obsCrossfilter } = this.props;
    if (obsCrossfilter.allSelectedLabels()) {
      dispatch(
        actions.requestEmbeddingLLMWithCells(obsCrossfilter.allSelectedLabels())
      );
    }
  };

  render() {
    const { outputText, loading, obsCrossfilter } = this.props;
    const { inputText } = this.state;

    return (
      <div
        style={{
          /* x y blur spread color */
          borderLeft: `1px solid ${globals.lightGrey}`,
          display: "flex",
          flexDirection: "row",
          justifyContent: "space-between",
          flexGrow: 1,
          position: "relative",
          overflowY: "inherit",
          height: "inherit",
          width: "inherit",
          padding: globals.leftSidebarSectionPadding,
        }}
      >
        <div style={{ flex: 2 }}>
          <InputGroup
            value={inputText}
            fill
            onChange={this.handleInputChange}
            placeholder="Type your message..."
          />
        </div>
        <Button
          onClick={this.findCellsClick}
          disabled={!inputText}
          loading={loading}
          style={{ margin: "0px 10px", padding: "0px 20px" }}
        >
          Find cells
        </Button>
        <Button
          onClick={this.describeSelectedClick}
          disabled={
            obsCrossfilter.countSelected() === 0 ||
            obsCrossfilter.countSelected() === obsCrossfilter.annoMatrix.nObs
          }
          loading={loading}
          style={{ margin: "0px 10px", padding: "0px 20px" }}
        >
          Describe selection
        </Button>
        <div
          style={{
            border: "1px solid #ccc",
            "max-height": "80px",
            "overflow-y": "auto",
            margin: "0px 5px",
            padding: "0px 5px",
            flex: 1,
          }}
        >
          <ul>
            {Object.keys(outputText).map((key) => (
              <li key={key}>
                <i>{key}</i>
                {Array.isArray(outputText[key])
                  ? renderList(outputText[key])
                  : `: ${outputText[key]}`}
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  }
}

export default BottomSideBar;
