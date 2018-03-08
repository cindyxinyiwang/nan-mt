folder_lists=(
  # "archived_exp/outputs_exp3_v2"
  # "archived_exp/outputs_exp3_v3"
  # "archived_exp/outputs_exp3_v4"
  # "archived_exp/outputs_exp3_v5"
  # "archived_exp/outputs_exp3_v6"
  # "archived_exp/outputs_exp3_v9"  # <-- Dev PPL 17.69
  # "archived_exp/outputs_exp3_v10"
  # "archived_exp/outputs_exp3_v11"
  # "outputs_exp4_v1"
  # "outputs_exp4_v2"
  "outputs_exp4_v6"
  "outputs_exp4_v7"
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

