import React from "react";
import { Button, Menu, MenuItem, Popover, Position } from "@blueprintjs/core";
import { IconNames } from "@blueprintjs/icons";

const InformationMenu = React.memo((props) => {
  const { libraryVersions, tosURL, privacyURL } = props;
  return (
    <Popover
      content={
        <Menu>
          <MenuItem
            href="https://www.biorxiv.org/content/10.1101/2024.10.15.618501v1"
            target="_blank"
            icon="document"
            text="CellWhisperer paper"
            rel="noopener"
          />
          <MenuItem
            href="https://docs.cellxgene.cziscience.com/"
            target="_blank"
            icon="manual"
            text="CELLxGENE docs"
            rel="noopener"
          />
          <MenuItem
            href="https://github.com/epigen/cellwhisperer/issues"
            target="_blank"
            icon="chat"
            text="Feedback (GitHub)"
            rel="noopener"
          />
          <MenuItem
            href="mailto:cellwhisperer@bocklab.org"
            icon="at"
            text="Feedback (Email)"
            rel="noopener"
          />
          <MenuItem
            href="https://github.com/epigen/cellwhisperer"
            target="_blank"
            icon="git-branch"
            text="Github"
            rel="noopener"
          />
          <MenuItem
            style={{ cursor: "default" }}
            target="_blank"
            text={libraryVersions?.cellxgene || null}
          />
          <MenuItem style={{ cursor: "default" }} text="MIT License" />
          {tosURL && (
            <MenuItem
              href={tosURL}
              target="_blank"
              text="Terms of Service"
              rel="noopener"
            />
          )}
          {privacyURL && (
            <MenuItem
              href={privacyURL}
              target="_blank"
              text="Privacy Policy"
              rel="noopener"
            />
          )}
        </Menu>
      }
      position={Position.BOTTOM_RIGHT}
      modifiers={{
        preventOverflow: { enabled: false },
        hide: { enabled: false },
      }}
    >
      <Button
        data-testid="menu"
        type="button"
        icon={IconNames.INFO_SIGN}
        style={{
          cursor: "pointer",
          verticalAlign: "middle",
        }}
      />
    </Popover>
  );
});

export default InformationMenu;
