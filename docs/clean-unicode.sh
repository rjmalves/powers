#!/bin/bash
# Replaces Unicode box-drawing and special characters with ASCII/LaTeX alternatives

sed -e 's/─/-/g' \
    -e 's/│/|/g' \
    -e 's/┌/+/g' \
    -e 's/┐/+/g' \
    -e 's/└/+/g' \
    -e 's/┘/+/g' \
    -e 's/├/+/g' \
    -e 's/┤/+/g' \
    -e 's/┬/+/g' \
    -e 's/┴/+/g' \
    -e 's/┼/+/g' \
    -e 's/▼/v/g' \
    -e 's/▶/>/g' \
    -e 's/►/>/g' \
    -e 's/₀/_0/g' \
    -e 's/₁/_1/g' \
    -e 's/₂/_2/g' \
    -e 's/₃/_3/g' \
    -e 's/₄/_4/g' \
    -e 's/₅/_5/g' \
    -e 's/₆/_6/g' \
    -e 's/₇/_7/g' \
    -e 's/₈/_8/g' \
    -e 's/₉/_9/g' \
    -e 's/ₜ/_t/g' \
    -e 's/ₕ/_h/g' \
    -e 's/ₖ/_k/g' \
    -e 's/ₘ/_m/g' \
    -e 's/ₙ/_n/g' \
    "$@"
