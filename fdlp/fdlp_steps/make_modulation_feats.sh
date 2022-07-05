#!/usr/bin/env bash

. ./path.sh

# Feature Options
nj=50
n_filters=20
fduration=1.5
frate=125
coeff_num=80
coeff_range='1,80'
order=80
overlap_fraction=0.50
lifter_file=
lfr=10
return_mvector=False
complex_mvectors=True
srate=16000
cmd=queue.pl
add_opts=

write_utt2num_frames=false
derivative_signal=false

conf_file=

. parse_options.sh || exit 1;

# Overwirte options from config file
if [ ! -z ${conf_file} ] ; then
  source ${conf_file}
fi

data_dir=$1
feat_dir=$2

echo "$0 $@"

# Convert feat_dir to the absolute file name

feat_dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir";} print $dir; ' $feat_dir ${PWD}`

mkdir -p $feat_dir

name=`basename $data_dir`
scp=$data_dir/wav.scp
log_dir=$data_dir/log
mkdir -p $log_dir
log_dir=`realpath ${log_dir}`

if ${write_utt2num_frames}; then
    add_opts="$add_opts --write_utt2num_frames"
fi

if [ ! -z ${lifter_file} ] ; then
  add_opts="$add_opts --lifter_file=${lifter_file}"
fi

# Split scp or segment files
if [ -f ${data_dir}/segments ]; then
  echo "$0 [info]: segments file exists: using that."
  split_segments=
  for n in $(seq $nj); do
    split_segments="$split_segments ${log_dir}/segments.$n"
  done

  utils/split_scp.pl ${data_dir}/segments ${split_segments} || exit 1;
else
   echo "$0 [info]: Splitting scp files for parallalization, no segment file found"
  split_scp=""
  for n in $(seq $nj); do
    split_scp="$split_scp $log_dir/wav_${name}.$n.scp"
  done
fi

  utils/split_scp.pl $scp $split_scp || exit 1;

echo "$0: Computing modulation features for scp files..."

# Compute modulation features

if [ -f ${data_dir}/segments ]; then

  $cmd --mem 5G JOB=1:$nj \
    $log_dir/feats_${name}.JOB.log \
    compute_modulation_features.py \
      $scp \
      $feat_dir/modspec_${name}.JOB \
      $add_opts \
      --segment_file=${log_dir}/segments.JOB \
      --n_filters=$n_filters \
      --fduration=$fduration \
      --frate=$frate \
      --coeff_num=${coeff_num} \
      --coeff_range=${coeff_range} \
      --order=$order \
      --overlap_fraction=${overlap_fraction} \
      --lfr=$lfr \
      --return_mvector=${return_mvector} \
      --complex_mvectors=${complex_mvectors} \
      --srate=$srate || exit 1;
else
    $cmd --mem 5G JOB=1:$nj \
    $log_dir/feats_${name}.JOB.log \
    compute_modulation_features.py \
      $log_dir/wav_${name}.JOB.scp \
      $feat_dir/modspec_${name}.JOB \
      $add_opts \
      --n_filters=$n_filters \
      --fduration=$fduration \
      --frate=$frate \
      --coeff_num=${coeff_num} \
      --coeff_range=${coeff_range} \
      --order=$order \
      --overlap_fraction=${overlap_fraction} \
      --lfr=$lfr \
      --return_mvector=${return_mvector} \
      --complex_mvectors=${complex_mvectors} \
      --srate=$srate || exit 1;

fi
  # concatenate all scp files together

  for n in $(seq $nj); do
    cat $feat_dir/modspec_$name.$n.scp || exit 1;
  done > $data_dir/feats.scp

  rm $log_dir/wav_${name}.*.scp

  # concatenate all length files together
  if $write_utt2num_frames; then
      for n in $(seq $nj); do
        cat $feat_dir/modspec_$name.$n.len || exit 1;
      done > $data_dir/utt2num_frames
  fi


echo $0": Finished computing mel spectrum features for $name"
