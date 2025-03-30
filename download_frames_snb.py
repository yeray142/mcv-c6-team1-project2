from huggingface_hub import snapshot_download

snapshot_download(repo_id="SoccerNet/SN-BAS-2025",
                  repo_type="dataset", revision="main",
                  local_dir="/ghome/c5mcv01/mcv-c6-team1-project2/data/soccernetball/SN-BAS-2025")