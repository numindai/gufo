{ dockerTools, cacert, gufo }:

# Use the production package and its runtime closure, without adding the
# development shell, source checkout or model weights.
dockerTools.streamLayeredImage {
  name = "gufo";
  tag = "build";
  contents = [ gufo cacert ];
  extraCommands = ''
    mkdir -p tmp
    chmod 1777 tmp
  '';
  config = {
    Entrypoint = [ "${gufo}/bin/gufo" ];
    Cmd = [ "--help" ];
    User = "1000:1000";
    Env = [
      "HOME=/tmp"
      "SSL_CERT_FILE=${cacert}/etc/ssl/certs/ca-bundle.crt"
    ];
    ExposedPorts."8080/tcp" = { };
    Labels = {
      "org.opencontainers.image.title" = "Gufo";
      "org.opencontainers.image.description" = "Gufo inference for AMD Strix Halo (gfx1151)";
      "org.opencontainers.image.licenses" = "MIT";
    };
  };
}
