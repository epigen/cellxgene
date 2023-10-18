import React from "react";
import { Button, InputGroup } from "@blueprintjs/core";
import * as globals from "../../globals";

class BottomSideBar extends React.Component {
  render() {
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
            value="Test"
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
          some text
        </div>
      </div>
    );
  }
}

export default BottomSideBar;
