version="cockroach-v19.2.6.linux-amd64"

cd /usr/local
    sudo curl --fail https://binaries.cockroachdb.com/$version.tgz | sudo tar -xz && sudo cp $version/cockroach bin/ || (echo "Failed to get CockroachDB binaries..." && exit 1)

    # some versions bring binaries of other libs that they rely on
    if [[ -d $version/lib ]]; then
        sudo mkdir -p lib/cockroach
        sudo cp $version/lib/libgeos.so lib/cockroach/
        sudo cp $version/lib/libgeos_c.so lib/cockroach/
    fi

cd - >> /dev/null

which cockroach

# to test cockroach installation
# cockroach demo
# SELECT ST_IsValid(ST_MakePoint(1,2));
