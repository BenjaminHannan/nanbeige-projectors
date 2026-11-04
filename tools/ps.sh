#!/bin/bash
# usage: ps.sh 'powershell code'   (or stdin with -)
if [ "$1" = "-" ]; then PS=$(cat); else PS="$1"; fi
ssh -o ConnectTimeout=20 benspc "powershell.exe -NoProfile -NonInteractive -EncodedCommand $(printf '%s' "$PS" | iconv -t UTF-16LE | base64)" 2>&1 | grep -v -e '#< CLIXML' -e '^<Objs'
