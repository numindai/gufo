{ lib, fetchurl }:
let
  dependencies = builtins.fromJSON (
    builtins.readFile ../../cmake/jinja-dependencies.json
  );
in
lib.mapAttrsToList (
  name: source:
  let archive = fetchurl (source // { name = "gufo-${lib.toLower name}.tar.gz"; });
  in "-DGUFO_${name}_URL=${archive}"
) dependencies
