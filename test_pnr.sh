ORIG_PROMPT="a DSLR photo of a man in a black tuxedo"
EDIT_PROMPT="a DSLR photo of a man in a black tuxedo, holding a champagne flute"

python launch.py --config configs/mvdream-sd21-shading-schedule.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$ORIG_PROMPT" \
    system.background.eval_color=[1.0,1.0,1.0] \
    tag="$ORIG_PROMPT" use_timestamp=false name="synthetic"

python launch.py --config configs/mvdream-pnr-synthetic.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$EDIT_PROMPT" \
    system.background.eval_color=[1.0,1.0,1.0] \
    system.weights="outputs/synthetic/$ORIG_PROMPT/ckpts/last.ckpt" \
    tag="$EDIT_PROMPT" use_timestamp=false name="pnr_synthetic"

python launch.py --config configs/mvdream-pnr-ipg.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$EDIT_PROMPT" \
    system.background.eval_color=[1.0,1.0,1.0] \
    system.weights="outputs/synthetic/$ORIG_PROMPT/ckpts/last.ckpt" \
    system.edit_weights="outputs/pnr_synthetic/$EDIT_PROMPT/ckpts/last.ckpt" \
    tag="$EDIT_PROMPT" use_timestamp=false name="pnr_synthetic_ipg"
