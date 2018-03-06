folder_lists=(
  "archived_exp/outputs_exp3_v2"
  # "archived_exp/outputs_exp3_v3"
  "archived_exp/outputs_exp3_v4"
  "outputs_exp3_v5"
  # "outputs_exp3_v6"
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

