import React from "react";
import { connect } from "react-redux";
import GeneExpression from "../geneExpression";
import ChatSideBar from "../chatSidebar";
import { H4 } from "@blueprintjs/core";
import * as globals from "../../globals";

@connect((state) => ({
  scatterplotXXaccessor: state.controls.scatterplotXXaccessor,
  scatterplotYYaccessor: state.controls.scatterplotYYaccessor,
}))
class RightSidebar extends React.Component {
  // Bar should capitalized...
  render() {
    return (
      <div
        style={{
          /* x y blur spread color */
          borderLeft: `1px solid ${globals.lightGrey}`,
          display: "flex",
          flexDirection: "column",
          position: "relative",
          overflowY: "inherit",
          height: "inherit",
          width: "inherit",
          padding: globals.leftSidebarSectionPadding,
        }}
      >
        <GeneExpression />

        <H4
          role="menuitem"
          tabIndex="0"
        >
          CellWhisperer
        </H4>
        <ChatSideBar />
      </div>
    );
  }
}

export default RightSidebar;
