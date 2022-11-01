import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import *

import gradio as gr
import gc

print(f'[INFO] loading options..')

# fake config object, this should not be used in CMD, only allow change from gradio UI.
parser = argparse.ArgumentParser()
parser.add_argument('--text', default=None, help="text prompt")
# parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
# parser.add_argument('-O2', action='store_true', help="equals --fp16 --dir_text")
parser.add_argument('--test', action='store_true', help="test mode")
parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
parser.add_argument('--workspace', type=str, default='trial_gradio')
parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
parser.add_argument('--seed', type=int, default=0)

### training options
parser.add_argument('--iters', type=int, default=10000, help="training iters")
parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
parser.add_argument('--ckpt', type=str, default='latest')
parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
parser.add_argument('--upsample_steps', type=int, default=64, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
# model options
parser.add_argument('--bg_radius', type=float, default=1.4, help="if positive, use a background model at sphere(bg_radius)")
parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
# network backbone
parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
parser.add_argument('--backbone', type=str, default='grid', help="nerf backbone, choose from [grid, tcnn, vanilla]")
# rendering resolution in training, decrease this if CUDA OOM.
parser.add_argument('--w', type=int, default=64, help="render width for NeRF in training")
parser.add_argument('--h', type=int, default=64, help="render height for NeRF in training")
parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")

### dataset options
parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
parser.add_argument('--fovy_range', type=float, nargs='*', default=[40, 70], help="training camera fovy range")
parser.add_argument('--dir_text', action='store_true', help="direction-encode the text prompt, by appending front/side/back/overhead view")
parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

parser.add_argument('--lambda_entropy', type=float, default=1e-4, help="loss scale for alpha entropy")
parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")

### GUI options
parser.add_argument('--gui', action='store_true', help="start a GUI")
parser.add_argument('--W', type=int, default=800, help="GUI width")
parser.add_argument('--H', type=int, default=800, help="GUI height")
parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
parser.add_argument('--fovy', type=float, default=60, help="default GUI camera fovy")
parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

opt = parser.parse_args() 

# default to use -O !!!
opt.fp16 = True
opt.dir_text = True
opt.cuda_ray = True
# opt.lambda_entropy = 1e-4
# opt.lambda_opacity = 0

if opt.backbone == 'vanilla':
    from nerf.network import NeRFNetwork
elif opt.backbone == 'tcnn':
    from nerf.network_tcnn import NeRFNetwork
elif opt.backbone == 'grid':
    from nerf.network_grid import NeRFNetwork
else:
    raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

print(opt)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'[INFO] loading models..')

if opt.guidance == 'stable-diffusion':
    from nerf.sd import StableDiffusion
    guidance = StableDiffusion(device)
elif opt.guidance == 'clip':
    from nerf.clip import CLIP
    guidance = CLIP(device)
else:
    raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()
test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()

print(f'[INFO] everything loaded!')

trainer = None
model = None

# define UI

# def load_mesh(mesh_file_name):
#     return [mesh_file_name, mesh_file_name]
#
# mesh_render = gr.Interface(
#         fn=load_mesh,
#         inputs=gr.Model3D(),
#         outputs=[
#             gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model"),
#             gr.File(label="Download 3D Model")
#         ],
#         examples=['/home/dingchaofan/stable-dreamfusion/trial_gradio/mesh/mesh.obj'], #  [os.path.join(os.path.dirname(__file__), "files/Bunny.obj")],
#         cache_examples=True,
#     )

with gr.Blocks(css=".gradio-container {max-width: 1024px; margin: auto;}") as demo:

    # title
    # gr.Markdown('[Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion) Text-to-3D Example')
    gr.Markdown(' 【文字转3D模型】 @Author : 丁超凡 （内测版本 并发数只能为一！）')

    # inputs
    prompt = gr.Textbox(label="输入希望生成的3D模型描述，目前仅支持英文", max_lines=1, value="a DSLR photo of a delicious hamburger")
    iters = gr.Slider(label="迭代次数，默认15000次", minimum=100, maximum=20000, value=15000, step=100)
    seed = gr.Slider(label="随机种子", minimum=0, maximum=2147483647, step=1, randomize=True)
    button = gr.Button('点击生成！')
    # mesh_button = gr.Button('Generate mesh')

    # outputs
    image = gr.Image(label="实时帧展示", visible=True)
    video = gr.Video(label="最终渲染视频", visible=False)
    logs = gr.Textbox(label="训练log")
    mesh = gr.Model3D(label="3D mesh模型展示", visible=True)
    file = gr.File(label="mesh文件下载", visible=False)


    # gradio main func
    def submit(text, iters, seed):

        global trainer, model

        # seed
        opt.seed = seed
        opt.text = text
        opt.iters = iters

        seed_everything(seed)

        # clean up
        if trainer is not None:
            del model
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            print('[INFO] clean up!')

        # simply reload everything...
        model = NeRFNetwork(opt)
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('df', opt, model, guidance, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        # train (every ep only contain 8 steps, so we can get some vis every ~10s)
        STEPS = 8
        max_epochs = np.ceil(opt.iters / STEPS).astype(np.int32)

        # we have to get the explicit training loop out here to yield progressive results...
        loader = iter(valid_loader)

        start_t = time.time()

        for epoch in range(max_epochs):

            trainer.train_gui(train_loader, step=STEPS)
            
            # manual test and get intermediate results
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(valid_loader)
                data = next(loader)

            trainer.model.eval()

            if trainer.ema is not None:
                trainer.ema.store()
                trainer.ema.copy_to()

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=trainer.fp16):
                    preds, preds_depth = trainer.test_step(data, perturb=False)

            if trainer.ema is not None:
                trainer.ema.restore()

            pred = preds[0].detach().cpu().numpy()
            # pred_depth = preds_depth[0].detach().cpu().numpy()

            pred = (pred * 255).astype(np.uint8)

            yield {
                image: gr.update(value=pred, visible=True),
                video: gr.update(visible=False),
                logs: f"training iters: {epoch * STEPS} / {iters}, lr: {trainer.optimizer.param_groups[0]['lr']:.6f}",
                mesh : gr.update(visible=True),
                file : gr.update(visible=False)
            }

        # test
        trainer.test(test_loader,write_video=False)

        trainer.save_mesh(resolution=128)  # Johnson
        mesh_result_wen = glob.glob(os.path.join(opt.workspace, 'mesh', 'mesh.obj'))
        print('mesh_result_wen=', mesh_result_wen)
        # mesh_result = os.path.join(opt.workspace, 'mesh', 'mesh.obj')
        mesh_result = '/home/dingchaofan/stable-dreamfusion/trial_gradio/mesh/mesh.obj'
        print('mesh_result ', mesh_result)

        # results = glob.glob(os.path.join(opt.workspace, 'results', '*_rgb.mp4'))
        # assert results is not None, "cannot retrieve results!"
        # results.sort(key=lambda x: os.path.getmtime(x)) # sort by mtime


        end_t = time.time()

        print('done ----------------')

        yield {
            image: gr.update(visible=False),
            # video: gr.update(value=results[-1], visible=True),
            video: gr.update(visible=False),
            logs: f"训练结束！共耗时 {(end_t - start_t)/ 60:.4f} 分钟! 请继续等待下方mesh模型展示加载完成，或直接下载文件 ",
            mesh: gr.update(value=mesh_result, visible=True),
            file: gr.update(value=mesh_result, visible=True)
        }


    button.click(
        submit, 
        inputs=[prompt, iters, seed],
        outputs=[image, video, logs, mesh, file]
    )

    # mesh_render.render()




# concurrency_count: only allow ONE running progress, else GPU will OOM.
demo.queue(concurrency_count=1)

demo.launch(debug=True, share=True, show_error=True)