import React from "react";
import { connect } from "react-redux";
import Categorical from "../categorical";
import * as globals from "../../globals";
import DynamicScatterplot from "../scatterplot/scatterplot";
import TopLeftLogoAndTitle from "./topLeftLogoAndTitle";
import Continuous from "../continuous/continuous";

@connect((state) => ({
  scatterplotXXaccessor: state.controls.scatterplotXXaccessor,
  scatterplotYYaccessor: state.controls.scatterplotYYaccessor,
  schema: state.annoMatrix?.schema,
}))
class LeftSideBar extends React.Component {

  constructor(props) {
    super(props);
    this.scrollContainerRef = React.createRef();
  }
  componentDidUpdate(prevProps) {
    // check for updates to the continuous fields. If a new one was added scroll down
    const { schema } = this.props;
    if (schema && prevProps.schema) {
      const prevContinuousNames = this.getContinuousNames(prevProps.schema);
      const currentContinuousNames = this.getContinuousNames(schema);

      if (currentContinuousNames.length > prevContinuousNames.length) {
        // There are more continuous items than before
        setTimeout(() => this.scrollToBottom(), 0);
      }
    }
  }

  getContinuousNames(schema) {
    const obsIndex = schema.annotations.obs.index;
    return schema.annotations.obs.columns
      .filter((col) => col.type === "int32" || col.type === "float32")
      .filter((col) => col.name !== obsIndex)
      .filter((col) => !col.writable)
      .map((col) => col.name);
  }

  scrollToBottom() {
    const scrollContainer = this.scrollContainerRef.current;
    if (scrollContainer) {

      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  }

  render() {
    const { scatterplotXXaccessor, scatterplotYYaccessor } = this.props;
    return (
      <div
        style={{
          /* x y blur spread color */
          borderRight: `1px solid ${globals.lightGrey}`,
          display: "flex",
          flexDirection: "column",
          height: "100%",
        }}
      >
        <TopLeftLogoAndTitle />
        <div
          ref={this.scrollContainerRef} // Attach the ref here
          style={{
            height: "100%",
            width: globals.leftSidebarWidth,
            overflowY: "auto",
          }}
        >
          <Categorical />
          <Continuous />
        </div>
        {scatterplotXXaccessor && scatterplotYYaccessor ? (
          <DynamicScatterplot />
        ) : null}
      </div>
    );
  }
}

export default LeftSideBar;
