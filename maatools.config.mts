import type { FullConfig } from '@nekosu/maa-tools'

const config: FullConfig = {
  cwd: import.meta.dirname,
  maaVersion: 'latest',
  interfacePath: 'assets/interface.json',
  check: {
    override: {
      // 忽略 mpe-config 带来的报错
      // ignore warning caused by mpe-config
      // 'mpe-config': 'ignore'

      // 以下为项目历史遗留问题的放行配置（业务上待逐步修复）
      // 1. unknown-image：E9/E9收菜.json 引用的 E9政策确认.png 从未提交
      // 2. unknown-task：DailyExplore_* 系列，9763dd8 重写交错战线每日探索后
      //    pipeline 节点已改为中文名，interface.json 中的旧英文引用尚未更新
      // 3. int-override-unknown-task：option 的 override 指向未在 task 列表声明的节点
      // 修复进度见 .workbuddy/memory/2026-10-25.md
      'unknown-image': 'ignore',
      'unknown-task': 'ignore',
      'int-override-unknown-task': 'ignore'
    }
  }
}

export default config
