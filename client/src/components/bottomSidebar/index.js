import React from "react";
import { connect } from "react-redux";
import { Button, InputGroup } from "@blueprintjs/core";
import actions from "../../actions";
import * as globals from "../../globals";

@connect((state) => state.llmEmbeddings)
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
    dispatch(actions.requestEmbeddingLLMWithTextAction(inputText));
  };

  describeSelect1Click = () => {
    const { dispatch } = this.props;
    dispatch(actions.requestEmbeddingLLMWithCellsAction());
  };

  render() {
    const { outputText } = this.props;
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
          style={{ margin: "0px 10px", padding: "0px 20px" }}
        >
          Find cells
        </Button>
        <Button
          onClick={this.describeSelect1Click}
          style={{ margin: "0px 10px", padding: "0px 20px" }}
        >
          Describe select 1
        </Button>
        <div
          style={{
            border: "1px solid #ccc",
            margin: "0px 10px",
            padding: "0px 10px",
            flex: 1,
          }}
        >
          {outputText}
        </div>
      </div>
    );
  }
}

export default BottomSideBar;
