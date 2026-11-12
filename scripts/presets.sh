# Flag sets for python -m blinkmimic2.train. The defaults of train.py are the BlinkMimic 2 recipe.
declare -A PRESET
PRESET[main]=""
# Same-time baseline of the paper: the same learner and code with the first-phase settings for the whole run (no exams,
# frontier practice, failure penalty, deployment stream or step schedule; the step falls by about 0.5% over the run).
PRESET[baseline]="--phase= --exam_frac 0 --exam_mix= --exam_pre_every 0 --exam_pre_near 0 --exam_pre_steps 500 --front_share 0 --start_prac_frac -1 --fail_penalty 0 --no-pen_horizon_only --no-reset_aprev_ref --no-phase_refill --share_ramp= --wstream_frac 0 --lr_decay_iters 250000 --lr_end 0.03 --iterations 7000 --ckpt_every 100"
