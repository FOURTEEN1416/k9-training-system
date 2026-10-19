<script setup lang="ts">
import { computed, h, onMounted, ref } from 'vue'
import {
  NAlert, NButton, NCard, NDataTable, NDivider, NEmpty, NGrid, NGridItem,
  NInput, NInputNumber, NModal, NSelect, NSpace, NSpin, NStatistic, NSwitch, NTag,
  NText, useMessage,
  type DataTableColumns, type SelectOption,
} from 'naive-ui'
import { CheckmarkOutline, RefreshOutline, PlayOutline, CloudDownloadOutline } from '@vicons/ionicons5'
import { request } from '@/api/http'
import { useAuthStore } from '@/stores/auth'

// ============================================================
// 角色（服务端 HandlerRead.role 序列化为小写，统一转大写比较）
// ============================================================
const auth = useAuthStore()
const role = computed(() => (auth.user?.role || '').toUpperCase())
const canReview = computed(() => ['ADMIN', 'MANAGER', 'RESEARCHER', 'HANDLER'].includes(role.value))
const canExport = computed(() => ['ADMIN', 'MANAGER'].includes(role.value))
const isViewer = computed(() => role.value === 'VIEWER')

const message = useMessage()

// ============================================================
// 类型
// ============================================================
interface QueueItem {
  task_id: number | null
  video_id: number
  video_filename: string
  scene: string
  status: string
  progress: number
  total_frames?: number | null
  annotated_frames: number
  episodes_total: number
  episodes_reviewed: number
  episodes_pending: number
  accepted: number
  rejected: number
  corrected: number
}

interface Episode {
  behavior_id: number
  video_id: number
  reviewed: boolean
  decision: string | null
  note: string | null
  reviewed_at: string | null
  original: { behavior: string; start_frame: number; end_frame: number }
  effective: { behavior: string; start_frame: number; end_frame: number }
  behavior: string
  start_frame: number
  end_frame: number
  confidence: number
  detector: string
  mappable: boolean
}

interface EpisodeResponse {
  video_id: number
  task_id: number | null
  task_status: string | null
  episodes: Episode[]
  mapped_behaviors: string[]
  unmappable_db_only: string[]
}

interface Summary {
  video_id: number
  task_id: number | null
  task_status: string | null
  total: number
  accepted: number
  rejected: number
  corrected: number
  pending: number
  annotated_frames: number
  progress: number
  mappable: number
  unmappable: number
}

// ============================================================
// 状态
// ============================================================
const loadingQueue = ref(false)
const loadingEpisodes = ref(false)
const saving = ref(false)
const queue = ref<QueueItem[]>([])
const selectedVideoId = ref<number | null>(null)
const episodes = ref<Episode[]>([])
const summary = ref<Summary | null>(null)
const unmappable = ref<string[]>([])
const mappedBehaviors = ref<string[]>([])

// 审核编辑弹窗
const reviewOpen = ref(false)
const reviewTarget = ref<Episode | null>(null)
const reviewDecision = ref<'accepted' | 'rejected' | 'corrected'>('accepted')
const reviewBehavior = ref<string | null>(null)
const reviewStart = ref<number | null>(null)
const reviewEnd = ref<number | null>(null)
const reviewNote = ref('')

// 导出/触发
const dataSource = ref<'synthetic' | 'real'>('synthetic')
const seed = ref(42)
const includeUnreviewed = ref(false)
const exports = ref<Array<{ export_id: string; created_at?: string; data_source?: string; stats?: Record<string, unknown> }>>([])
const selectedExportId = ref<string | null>(null)
const triggerEpochs = ref(20)
const triggerBatch = ref(8)
const taskQueryId = ref('')
const taskResult = ref<Record<string, unknown> | null>(null)

const behaviorOptions = computed<SelectOption[]>(() =>
  mappedBehaviors.value.map(b => ({ label: b, value: b })),
)

// ============================================================
// 数据加载
// ============================================================
async function loadQueue() {
  loadingQueue.value = true
  try {
    const resp = await request<{ items: QueueItem[]; total: number }>({
      url: '/api/annotations/review/queue',
      params: { limit: 100 },
    })
    queue.value = resp.items
    if (selectedVideoId.value == null && resp.items.length > 0) {
      await selectVideo(resp.items[0].video_id)
    }
  } catch (e) {
    message.error(`加载审核队列失败: ${(e as Error).message}`)
  } finally {
    loadingQueue.value = false
  }
}

async function selectVideo(videoId: number) {
  selectedVideoId.value = videoId
  await Promise.all([loadEpisodes(videoId), loadSummary(videoId)])
}

async function loadEpisodes(videoId: number) {
  loadingEpisodes.value = true
  try {
    const resp = await request<EpisodeResponse>({
      url: `/api/annotations/videos/${videoId}/episodes`,
    })
    episodes.value = resp.episodes
    mappedBehaviors.value = resp.mapped_behaviors
    unmappable.value = resp.unmappable_db_only
  } catch (e) {
    message.error(`加载 episode 失败: ${(e as Error).message}`)
    episodes.value = []
  } finally {
    loadingEpisodes.value = false
  }
}

async function loadSummary(videoId: number) {
  try {
    summary.value = await request<Summary>({
      url: `/api/annotations/videos/${videoId}/review-summary`,
    })
  } catch (e) {
    message.error(`加载审核统计失败: ${(e as Error).message}`)
    summary.value = null
  }
}

// ============================================================
// 审核
// ============================================================
function openReview(ep: Episode, decision: 'accepted' | 'rejected' | 'corrected') {
  if (!canReview.value) {
    message.warning('当前角色无审核权限（服务端 403 为准）')
    return
  }
  reviewTarget.value = ep
  reviewDecision.value = decision
  reviewBehavior.value = ep.effective.behavior
  reviewStart.value = ep.effective.start_frame
  reviewEnd.value = ep.effective.end_frame
  reviewNote.value = ''
  reviewOpen.value = true
}

async function submitReview() {
  const ep = reviewTarget.value
  if (!ep) return
  if (!ep.mappable) {
    message.error('该行为类不在 16 类可审核标签空间内（管线未产出）')
    return
  }
  saving.value = true
  try {
    const body: Record<string, unknown> = {
      decision: reviewDecision.value,
      note: reviewNote.value || undefined,
    }
    if (reviewDecision.value === 'corrected') {
      if (reviewBehavior.value && reviewBehavior.value !== ep.original.behavior) {
        body.behavior = reviewBehavior.value
      }
      if (reviewStart.value != null && reviewStart.value !== ep.original.start_frame) {
        body.start_frame = reviewStart.value
      }
      if (reviewEnd.value != null && reviewEnd.value !== ep.original.end_frame) {
        body.end_frame = reviewEnd.value
      }
      if (body.behavior === undefined && body.start_frame === undefined && body.end_frame === undefined) {
        message.warning('改类/改边界至少改动一项')
        saving.value = false
        return
      }
    }
    await request({
      url: `/api/annotations/episodes/${ep.behavior_id}/review`,
      method: 'POST',
      data: body,
    })
    message.success('审核已保存')
    reviewOpen.value = false
    if (selectedVideoId.value != null) {
      await Promise.all([loadEpisodes(selectedVideoId.value), loadSummary(selectedVideoId.value)])
      await loadQueue()
    }
  } catch (e) {
    message.error(`保存审核失败: ${(e as Error).message}`)
  } finally {
    saving.value = false
  }
}

async function completeTask() {
  const taskId = summary.value?.task_id
  if (taskId == null) {
    message.warning('当前视频尚无标注任务')
    return
  }
  saving.value = true
  try {
    await request({
      url: `/api/annotations/tasks/${taskId}/complete`,
      method: 'POST',
    })
    message.success('任务已完成')
    if (selectedVideoId.value != null) {
      await Promise.all([loadEpisodes(selectedVideoId.value), loadSummary(selectedVideoId.value)])
    }
    await loadQueue()
  } catch (e) {
    message.error(`完成任务失败: ${(e as Error).message}`)
  } finally {
    saving.value = false
  }
}

// ============================================================
// 导出 / 触发
// ============================================================
async function runExport() {
  if (!canExport.value) {
    message.warning('导出需要 ADMIN/MANAGER 角色')
    return
  }
  saving.value = true
  try {
    const result = await request<{ export_id: string }>({
      url: '/api/finetune/export',
      method: 'POST',
      data: {
        include_unreviewed: includeUnreviewed.value,
        data_source: dataSource.value,
        seed: seed.value,
      },
    })
    message.success(`导出完成: ${result.export_id}`)
    await loadExports()
    selectedExportId.value = result.export_id
  } catch (e) {
    message.error(`导出失败: ${(e as Error).message}`)
  } finally {
    saving.value = false
  }
}

async function loadExports() {
  try {
    const resp = await request<{ items: Array<{ export_id: string; created_at?: string; data_source?: string; stats?: Record<string, unknown> }> }>({
      url: '/api/finetune/exports',
    })
    exports.value = resp.items
    if (!selectedExportId.value && resp.items.length > 0) {
      selectedExportId.value = resp.items[0].export_id
    }
  } catch (e) {
    // MANAGER/RESEARCHER 之外的 403 等：不打断页面
    message.error(`加载导出记录失败: ${(e as Error).message}`)
  }
}

async function triggerFinetune() {
  if (!canExport.value) {
    message.warning('触发微调需要 ADMIN/MANAGER 角色')
    return
  }
  if (!selectedExportId.value) {
    message.warning('请先选择导出记录')
    return
  }
  saving.value = true
  try {
    const result = await request<{ task_id: string }>({
      url: '/api/finetune/trigger',
      method: 'POST',
      data: {
        export_id: selectedExportId.value,
        epochs: triggerEpochs.value,
        batch_size: triggerBatch.value,
      },
    })
    message.success(`微调已派发: ${result.task_id}`)
    taskQueryId.value = result.task_id
  } catch (e) {
    message.error(`触发微调失败: ${(e as Error).message}`)
  } finally {
    saving.value = false
  }
}

async function queryTask() {
  if (!taskQueryId.value) {
    message.warning('请输入 task_id')
    return
  }
  try {
    taskResult.value = await request<Record<string, unknown>>({
      url: `/api/finetune/tasks/${taskQueryId.value}`,
    })
  } catch (e) {
    message.error(`查询任务失败: ${(e as Error).message}`)
    taskResult.value = null
  }
}

// ============================================================
// 表格
// ============================================================
const queueColumns: DataTableColumns<QueueItem> = [
  { title: '任务', key: 'task_id', width: 70, render: row => row.task_id ?? '—' },
  { title: '视频', key: 'video_filename', ellipsis: { tooltip: true } },
  {
    title: '状态', key: 'status', width: 100,
    render: row => h(
      NTag,
      { size: 'small', type: row.status === 'no_task' ? 'warning' : 'default' },
      { default: () => (row.status === 'no_task' ? '待创建' : row.status) },
    ),
  },
  {
    title: '审核进度', key: 'episodes_reviewed', width: 110,
    render: row => `${row.episodes_reviewed}/${row.episodes_total}`,
  },
  {
    title: '操作', key: 'actions', width: 90,
    render: row => h(NButton, {
      size: 'small', type: selectedVideoId.value === row.video_id ? 'primary' : 'default',
      onClick: () => selectVideo(row.video_id),
    }, { default: () => '选择' }),
  },
]

const episodeColumns: DataTableColumns<Episode> = [
  { title: 'ID', key: 'behavior_id', width: 60 },
  { title: '原类', key: 'original', width: 100, render: row => row.original.behavior },
  {
    title: '原边界', key: 'bounds', width: 110,
    render: row => `${row.original.start_frame}–${row.original.end_frame}`,
  },
  {
    title: '生效类', key: 'behavior', width: 100,
    render: row => row.reviewed ? row.effective.behavior : '—',
  },
  {
    title: '置信度', key: 'confidence', width: 90,
    render: row => row.confidence.toFixed(2),
  },
  {
    title: '审核态', key: 'decision', width: 100,
    render: row => h(
      NTag,
      { size: 'small', type: row.decision === 'rejected' ? 'error' : row.decision ? 'success' : 'warning' },
      { default: () => row.decision || '待审' },
    ),
  },
  {
    title: '操作', key: 'actions', width: 220,
    render: row => h(NSpace, { size: 4 }, {
      default: () => [
        h(NButton, {
          size: 'tiny', type: 'success', disabled: !canReview.value || !row.mappable,
          onClick: () => openReview(row, 'accepted'),
        }, { default: () => '接受' }),
        h(NButton, {
          size: 'tiny', type: 'error', disabled: !canReview.value || !row.mappable,
          onClick: () => openReview(row, 'rejected'),
        }, { default: () => '拒绝' }),
        h(NButton, {
          size: 'tiny', disabled: !canReview.value || !row.mappable,
          onClick: () => openReview(row, 'corrected'),
        }, { default: () => '改类/改边界' }),
      ],
    }),
  },
]

onMounted(async () => {
  if (!auth.user) {
    try {
      await auth.fetchMe()
    } catch {
      // 401 由 http 拦截器统一处理
    }
  }
  await loadQueue()
  if (canExport.value) {
    await loadExports()
  }
})
</script>

<template>
  <div style="padding: 16px; max-width: 1400px; margin: 0 auto;">
    <NAlert type="warning" :bordered="false" style="margin-bottom: 12px;">
      合成数据仅用于工程验收，工程验收 ≠ 生产验收；不代表识别精度或生产可用。
      模型注册后 is_active=False，激活需人工执行 POST /api/models/{id}/activate。
    </NAlert>

    <NAlert v-if="isViewer" type="info" :bordered="false" style="margin-bottom: 12px;">
      VIEWER 角色只读：可查看审核队列与 episode，审核/导出/触发按钮不可用（服务端 403 为准）。
    </NAlert>

    <NCard title="标注审核" size="small" style="margin-bottom: 16px;">
      <template #header-extra>
        <NButton size="small" :loading="loadingQueue" @click="loadQueue">
          <template #icon><RefreshOutline /></template>
          刷新队列
        </NButton>
      </template>

      <NSpin :show="loadingQueue">
        <NDataTable
          v-if="queue.length > 0"
          :columns="queueColumns"
          :data="queue"
          :row-key="(row: QueueItem) => `${row.video_id}-${row.task_id ?? 'none'}`"
          size="small"
          :max-height="240"
        />
        <NEmpty v-else description="暂无待审核任务（正常 pipeline 视频首次审核会自动创建任务）" />
      </NSpin>
    </NCard>

    <NCard v-if="selectedVideoId != null" title="Episode 审核" size="small" style="margin-bottom: 16px;">
      <template #header-extra>
        <NSpace>
          <NButton size="small" type="primary" :loading="saving" :disabled="!canReview" @click="completeTask">
            <template #icon><CheckmarkOutline /></template>
            完成任务
          </NButton>
        </NSpace>
      </template>

      <NGrid :cols="6" :x-gap="12" style="margin-bottom: 12px;">
        <NGridItem><NStatistic size="small" label="总数" :value="summary?.total ?? 0" /></NGridItem>
        <NGridItem><NStatistic size="small" label="接受" :value="summary?.accepted ?? 0" /></NGridItem>
        <NGridItem><NStatistic size="small" label="拒绝" :value="summary?.rejected ?? 0" /></NGridItem>
        <NGridItem><NStatistic size="small" label="改类" :value="summary?.corrected ?? 0" /></NGridItem>
        <NGridItem><NStatistic size="small" label="待审" :value="summary?.pending ?? 0" /></NGridItem>
        <NGridItem>
          <NStatistic
            size="small"
            label="已覆盖帧"
            :value="summary?.annotated_frames ?? 0"
          />
        </NGridItem>
      </NGrid>

      <NAlert v-if="unmappable.length > 0" type="error" :bordered="false" style="margin-bottom: 12px;">
        管线未产出，不可审核（DB-only P2 类）: {{ unmappable.join(', ') }}
      </NAlert>

      <NSpin :show="loadingEpisodes">
        <NDataTable
          v-if="episodes.length > 0"
          :columns="episodeColumns"
          :data="episodes"
          :row-key="(row: Episode) => row.behavior_id"
          size="small"
          :max-height="420"
        />
        <NEmpty v-else description="该视频暂无可审核的 behavior episode" />
      </NSpin>
    </NCard>

    <NCard v-if="canExport" title="导出与微调（工程验收）" size="small">
      <NGrid :cols="4" :x-gap="12" style="margin-bottom: 12px;">
        <NGridItem>
          <NText depth="3">数据来源</NText>
          <NSelect
            v-model:value="dataSource"
            size="small"
            :options="[
              { label: 'synthetic（合成，仅工程验收）', value: 'synthetic' },
              { label: 'real（真实数据）', value: 'real' },
            ]"
          />
        </NGridItem>
        <NGridItem>
          <NText depth="3">划分种子</NText>
          <NInputNumber v-model:value="seed" size="small" :min="0" />
        </NGridItem>
        <NGridItem>
          <NText depth="3">包含未审核</NText>
          <NSwitch v-model:value="includeUnreviewed" size="small" />
          <NText depth="3" style="margin-left: 6px;">{{ includeUnreviewed ? '是（仅工程调试）' : '否（默认）' }}</NText>
        </NGridItem>
        <NGridItem>
          <NButton type="primary" size="small" :loading="saving" @click="runExport" style="margin-top: 20px;">
            <template #icon><CloudDownloadOutline /></template>
            导出 16 类数据集
          </NButton>
        </NGridItem>
      </NGrid>

      <NDivider style="margin: 8px 0;" />

      <NGrid :cols="5" :x-gap="12">
        <NGridItem>
          <NText depth="3">导出记录</NText>
          <NSelect
            v-model:value="selectedExportId"
            size="small"
            :options="exports.map(e => ({ label: e.export_id, value: e.export_id }))"
          />
        </NGridItem>
        <NGridItem>
          <NText depth="3">epochs</NText>
          <NInputNumber v-model:value="triggerEpochs" size="small" :min="1" :max="200" />
        </NGridItem>
        <NGridItem>
          <NText depth="3">batch_size</NText>
          <NInputNumber v-model:value="triggerBatch" size="small" :min="1" :max="64" />
        </NGridItem>
        <NGridItem>
          <NButton type="primary" size="small" :loading="saving" @click="triggerFinetune" style="margin-top: 20px;">
            <template #icon><PlayOutline /></template>
            触发 CPU 微调
          </NButton>
        </NGridItem>
        <NGridItem>
          <NText depth="3">查询 task_id</NText>
          <NSpace>
            <NInput v-model:value="taskQueryId" size="small" placeholder="Celery task id" />
            <NButton size="small" @click="queryTask">查询</NButton>
          </NSpace>
        </NGridItem>
      </NGrid>

      <pre v-if="taskResult" style="margin-top: 12px; font-size: 12px; overflow: auto;">{{ JSON.stringify(taskResult, null, 2) }}</pre>
    </NCard>

    <NModal
      v-model:show="reviewOpen"
      preset="card"
      :title="`审核 episode #${reviewTarget?.behavior_id ?? ''}`"
      style="width: 520px;"
    >
      <NSpace vertical size="medium">
        <NText depth="3">
          原值: {{ reviewTarget?.original.behavior }}（{{ reviewTarget?.original.start_frame }}–{{ reviewTarget?.original.end_frame }}）
        </NText>
        <NSelect
          v-model:value="reviewDecision"
          :options="[
            { label: '接受', value: 'accepted' },
            { label: '拒绝', value: 'rejected' },
            { label: '改类/改边界', value: 'corrected' },
          ]"
        />
        <template v-if="reviewDecision === 'corrected'">
          <NSelect v-model:value="reviewBehavior" :options="behaviorOptions" placeholder="生效行为（16 类）" />
          <NSpace>
            <NInputNumber v-model:value="reviewStart" :min="0" placeholder="start_frame" />
            <NInputNumber v-model:value="reviewEnd" :min="0" placeholder="end_frame" />
          </NSpace>
        </template>
        <NInput v-model:value="reviewNote" type="textarea" placeholder="备注（可选）" />
        <NSpace justify="end">
          <NButton @click="reviewOpen = false">取消</NButton>
          <NButton type="primary" :loading="saving" @click="submitReview">保存</NButton>
        </NSpace>
      </NSpace>
    </NModal>
  </div>
</template>
