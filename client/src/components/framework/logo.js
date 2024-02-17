import React from "react";

const Logo = (props) => {
  const { size, alt, src } = props;
  return <img src={src} height={size} width={size} alt={alt} />;
};

export default Logo;
