folder_lists=(
  # "archived_exp/outputs_exp3_v2"
  # "archived_exp/outputs_exp3_v3"
  # "archived_exp/outputs_exp3_v4"
  # "archived_exp/outputs_exp3_v5"
  # "archived_exp/outputs_exp3_v6"
  # "archived_exp/outputs_exp3_v9"
  # "archived_exp/outputs_exp3_v10"
  # "archived_exp/outputs_exp3_v11"
  # "archived_exp/outputs_exp4_v1"
  # "archived_exp/outputs_exp4_v2"
  # "archived_exp/outputs_exp4_v6"  # <-- Dev PPL: 14.63. Dev BLEU: 11.01
  # "archived_exp/outputs_exp4_v7"
  # "archived_exp/outputs_exp4_v9"
  # "archived_exp/outputs_exp5_v1"
  # "archived_exp/outputs_exp5_v2"
  # "archived_exp/outputs_exp5_v3"  # <-- Dev PPL: 12.61. Dev BLEU: 15.16
  # "archived_exp/outputs_exp5_v6"
  # "archived_exp/outputs_exp5_v7"
  # "archived_exp/outputs_exp6_v1"
  # "archived_exp/outputs_exp6_v2"  # Dev PPL: 11.09. Dev BLEU: 22.89
  "outputs_exp6_v3"
  # "outputs_exp6_v4"
  "outputs_exp6_v5"
  # "outputs_exp6_v6"
  "outputs_exp6_v9"
)

folder_names=""

for (( i=0; i<${#folder_lists[@]}; i++ ));
do
  if [ $i -eq 0 ]
  then
    folder_names="${folder_lists[$i]}"
  else
    folder_names="$folder_names,${folder_lists[$i]}"
  fi
done

python visualize/plot_acc_curve.py \
  --output_dir="${pwd}" \
  --folder_names="$folder_names" \
  --plot_name="IWSLT 2016 dev2010 ppl" \
  "$@"

python visualize/plot_acc_curve.py \
  --output_dir="${pwd}" \
  --folder_names="$folder_names" \
  --plot_name="IWSLT 2016 dev2010 acc" \
  "$@"

