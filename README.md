# Hackathon catch — 本地代码快照

保存于 2026-09-19。包含当前已部署的持续 rough 接球逻辑、默认归位姿态、cam2 标定、配置、相关测试和原始标定数据。

## 机器上运行

`catch-plane.sh` 使用机器上的绝对路径 `/opt/viam/trajectory-local`、Python 环境 `/opt/viam/trajectory-local-venv` 和机器已有凭据。此快照没有复制 `.env`、云端凭据、虚拟环境或 Git 历史。

```bash
viam machine part shell --part 49d63d4f-4191-4d94-9e43-e379f846dd0e
cd /opt/viam/trajectory-local
./catch-plane.sh --trajectory-source wrist --rough --execute
```

启动会移动到 `motion/catch_plane.py` 内用户指定的 `DEFAULT_POSE`。每次尝试后等待落球，再归位继续待命。Ctrl+C 请求停止。去掉 `--execute` 仅预览，不移动、不归位。

## 当前策略与限制

- wrist 预测落点，cam2 判断接近；两份证据有效期均为 0.25 秒，提交前重新计算拦截。
- 单次平移及相对本次启动位置的平移均不超过 150 mm；保持碗高度与姿态。
- rough 允许迟到；不代表每次抛球都会触发。归位或运动失败会终止。
- cam2 仅作接近判定和预览，标定仍未通过独立验证，不能直接执行 side/stereo 接球。
- 当前仍存在 cam2 确认较晚、wrist 丢球或预测缓存失效导致不触发的问题；不是已解决的最终接球系统。
- `MOVE RESULT` 的目标误差是机械臂到达指令位置的误差，不是球的预测误差。

## 本地测试

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest tests.test_default_return tests.test_rough_cycle tests.test_ballistic tests.test_catch_side tests.test_stereo_tracking
```

以上相关测试 68 项通过；未声称全部仓库测试通过。硬件运行默认连接机器本机 `127.0.0.1:8080`，不能直接在笔记本上用原脚本控制远程机械臂。

`catch_plane.config.json` 从机器读取；`calibration_data/` 保存本次标定数据，便于后续排查。
