$encoders = @("hash", "predictive", "e0", "e1a", "e1c")
$streams = @("same-token-context", "role-sharing", "delayed-role")
$seeds = @(1, 2)

foreach ($enc in $encoders) {
    foreach ($str in $streams) {
        foreach ($seed in $seeds) {
            $key = "$enc/$str/$seed"
            Write-Host "=== $key ==="
            $json = & cargo run --release -- run e1a --stream $str --encoder $enc --steps 10000 --seed $seed 2> $null
            try {
                $d = $json | ConvertFrom-Json
                Write-Host "dense_CPI=$($d.dense_cpi) embed_sep=$($d.embedding_role_separation) feat_CPI=$($d.controlled_feature_predictive_info) dense_nll=$($d.dense_nll)"
                if ($d.attention_mass_base -ne $null) {
                    Write-Host "  mass_base=$($d.attention_mass_base) mass_proto=$($d.attention_mass_proto) top_c1=$($d.top_credit_1) attn_corr=$($d.attention_credit_corr) cpi_wo1=$($d.dense_cpi_without_top1)"
                }
            } catch {
                Write-Host "  PARSE ERROR: $_"
                Write-Host "  RAW: $json"
            }
        }
    }
}
