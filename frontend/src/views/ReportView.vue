<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  NCard, NSpace, NText, NEmpty, NStatistic, NTag, NButton, NIcon, NSpin,
  NDescriptions, NDescriptionsItem, NGrid, NGridItem, NAlert, NProgress,
  useMessage,
} from 'naive-ui'
import {
  DownloadOutline, ArrowBackOutline, RefreshOutline,
} from '@vicons/ionicons5'
import { api, type Video, type Score } from '@/api'

const route = useRoute()
const router = useRouter()
const message = useMessage()

const videoId = computed(() => {
  const v = route.params.id
  return v ? Number(v) : null
})

// === 数据 ===
const video = ref<Video | null>(null)
const scores = ref<Score[]>([])
const loading = ref(false)

// === 状态映射 ===
const statusTagType: Record<string, 'default' | 'info' | 'success' | 'error' | 'warning'> = {
  uploaded: 'default',
  processing: 'info',
  completed: 'success',
  failed: 'error',
}

const statusText: Record<string, string> = {
  uploaded: '已上传',
  processing: '推理中',
  completed: '已完成',
  failed: '失败',
}

const sceneText: Record<string, string> = {
  obedience_trial: '科目测评',
  puppy_selection: '幼犬选育',
}

// === 加载 ===
async function loadData() {
  if (videoId.value == null) return
  loading.value = true
  try {
    const [v, s] = await Promise.all([
      api.getVideo(videoId.value).catch(e => {
        message.error(`加载视频失败: ${(e as Error).message}`)
        return null
      }),
      api.listScores(videoId.value).catch(e => {
        message.error(`加载评分失败: ${(e as Error).message}`)
        return [] as Score[]
      }),
    ])
    video.value = v
    scores.value = s
  } finally {
    loading.value = false
  }
}

watch(videoId, () => loadData(), { immediate: false })

// === 派生 ===
const latestScore = computed<Score | null>(() =>
  scores.value.length > 0 ? scores.value[0] : null,
)

const canDownloadPdf = computed(() =>
  video.value?.status === 'completed' && !!video.value?.report_path,
)

const pdfUrl = computed(() =>
  videoId.value ? api.getReportUrl(videoId.value) : null,
)

// === 评分维度（取最新一条 score 的非空字段）===
interface Dimension {
  label: string
  value: number
  color: string
}

const dimensions = computed<Dimension[]>(() => {
  if (!latestScore.value) return []
  const s = latestScore.value
  const result: Dimension[] = []

  // 7 维评分（按场景过滤）
  const allDims: Array<[string, number | null | undefined, string]> = [
    ['准确度', s.accuracy, '#2080f0'],
    ['响应延迟', s.response_latency, '#f0a020'],
    ['保持时长', s.duration, '#18a058'],
    ['搜索效率', s.search_efficiency, '#d03050'],
    ['注意力', s.attention, '#7c3aed'],
    ['胆量欲望', s.courage, '#0ea5e9'],
    ['步态质量', s.gait_quality, '#10b981'],
  ]

  for (const [label, value, color] of allDims) {
    if (value != null) {
      result.push({ label, value: Math.round(value * 10) / 10, color })
    }
  }
  return result
})

// === 工具 ===
function formatTime(iso?: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('zh-CN', { hour12: false })
}

function formatDuration(sec?: number | null): string {
  if (sec == null) return '—'
  if (sec < 60) return `${sec.toFixed(1)} 秒`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec % 60)
  return `${m} 分 ${s} 秒`
}

function formatResolution(w?: number | null, h?: number | null): string {
  if (w == null || h == null) return '—'
  return `${w} × ${h}`
}

// === 操作 ===
function downloadPdf() {
  if (pdfUrl.value) window.open(pdfUrl.value, '_blank')
}

function goBack() {
  router.push('/history')
}

onMounted(() => loadData())
</script>

<template>
  <NSpace vertical :size="16">
    <!-- 头部 -->
    <NCard size="small" bordered>
      <NSpace align="center" justify="space-between">
        <NSpace align="center" :size="12">
          <NButton quaternary circle @click="goBack">
            <template #icon>
              <NIcon :component="ArrowBackOutline" />
            </template>
          </NButton>
          <NText strong style="font-size: 16px;">
            评分报告
            <template v-if="video">
              · {{ video.original_filename }}
            </template>
          </NText>
        </NSpace>
        <NSpace :size="8">
          <NButton size="small" quaternary @click="loadData">
            <template #icon>
              <NIcon :component="RefreshOutline" />
            </template>
            刷新
          </NButton>
          <NButton
            v-if="canDownloadPdf"
            size="small"
            type="primary"
            @click="downloadPdf"
          >
            <template #icon>
              <NIcon :component="DownloadOutline" />
            </template>
            下载 PDF
          </NButton>
        </NSpace>
      </NSpace>
    </NCard>

    <NSpin :show="loading">
      <!-- 无 ID -->
      <NCard v-if="videoId == null" size="medium" bordered>
        <NEmpty description="未指定视频 ID">
          <template #extra>
            <NButton type="primary" @click="goBack">查看历史记录</NButton>
          </template>
        </NEmpty>
      </NCard>

      <!-- 视频不存在 -->
      <NCard v-else-if="!loading && !video" size="medium" bordered>
        <NEmpty description="视频不存在或加载失败">
          <template #extra>
            <NButton type="primary" @click="goBack">返回历史记录</NButton>
          </template>
        </NEmpty>
      </NCard>

      <template v-else-if="video">
        <!-- 视频元数据 -->
        <NCard title="视频信息" size="medium" bordered>
          <NDescriptions :column="3" label-placement="left" bordered>
            <NDescriptionsItem label="视频 ID">{{ video.id }}</NDescriptionsItem>
            <NDescriptionsItem label="场景">
              <NTag size="small" round type="info">
                {{ sceneText[video.scene] || video.scene }}
              </NTag>
            </NDescriptionsItem>
            <NDescriptionsItem label="状态">
              <NTag :type="statusTagType[video.status] || 'default'" size="small" round>
                {{ statusText[video.status] || video.status }}
              </NTag>
            </NDescriptionsItem>
            <NDescriptionsItem label="文件名" :span="3">
              {{ video.original_filename }}
            </NDescriptionsItem>
            <NDescriptionsItem label="上传时间">{{ formatTime(video.uploaded_at) }}</NDescriptionsItem>
            <NDescriptionsItem label="处理完成时间">{{ formatTime(video.processed_at) }}</NDescriptionsItem>
            <NDescriptionsItem label="耗时">{{ formatDuration(video.duration_sec) }}</NDescriptionsItem>
            <NDescriptionsItem label="分辨率">{{ formatResolution(video.width, video.height) }}</NDescriptionsItem>
            <NDescriptionsItem label="FPS">{{ video.fps != null ? video.fps.toFixed(2) : '—' }}</NDescriptionsItem>
            <NDescriptionsItem label="犬只 ID">{{ video.dog_id ?? '—' }}</NDescriptionsItem>
            <NDescriptionsItem v-if="video.error_message" label="错误信息" :span="3">
              <NText type="error">{{ video.error_message }}</NText>
            </NDescriptionsItem>
          </NDescriptions>
        </NCard>

        <!-- 处理中 -->
        <NCard v-if="video.status === 'processing' || video.status === 'uploaded'" size="medium" bordered>
          <NAlert type="info" title="推理进行中">
            视频正在后台处理，请稍候片刻后刷新页面查看结果。
          </NAlert>
        </NCard>

        <!-- 失败 -->
        <NCard v-if="video.status === 'failed'" size="medium" bordered>
          <NAlert type="error" title="推理失败">
            {{ video.error_message || '未知错误' }}
          </NAlert>
        </NCard>

        <!-- 评分结果 -->
        <NCard v-if="latestScore" title="评分结果" size="medium" bordered>
          <template #header-extra>
            <NText depth="3" style="font-size: 12px;">
              评分标准: {{ latestScore.standard }} · 引擎版本: {{ latestScore.scoring_engine_version }}
            </NText>
          </template>

          <!-- 总分 -->
          <NGrid :cols="4" :x-gap="16" :y-gap="16" style="margin-bottom: 24px;">
            <NGridItem>
              <NStatistic label="综合分" :value="latestScore.overall">
                <template #suffix>
                  <NText depth="3" style="font-size: 14px;"> / 100</NText>
                </template>
              </NStatistic>
            </NGridItem>
            <NGridItem>
              <NStatistic label="评分 ID" :value="latestScore.id" />
            </NGridItem>
            <NGridItem>
              <NStatistic label="评分时间" :value="formatTime(latestScore.created_at)" />
            </NGridItem>
            <NGridItem>
              <NStatistic
                label="判定"
                :value="latestScore.overall >= 70 ? '合格' : latestScore.overall >= 60 ? '基本合格' : '不合格'"
              >
                <template #default>
                  <NTag
                    :type="latestScore.overall >= 70 ? 'success' : latestScore.overall >= 60 ? 'warning' : 'error'"
                    size="large"
                    round
                  >
                    {{ latestScore.overall >= 70 ? '合格' : latestScore.overall >= 60 ? '基本合格' : '不合格' }}
                  </NTag>
                </template>
              </NStatistic>
            </NGridItem>
          </NGrid>

          <!-- 各维度评分 -->
          <NText strong style="display: block; margin-bottom: 12px;">维度评分</NText>
          <NSpace vertical :size="12">
            <div v-for="dim in dimensions" :key="dim.label">
              <NSpace align="center" justify="space-between" style="margin-bottom: 4px;">
                <NText>{{ dim.label }}</NText>
                <NText strong :style="{ color: dim.color }">{{ dim.value }}</NText>
              </NSpace>
              <NProgress
                type="line"
                :percentage="dim.value"
                :show-indicator="false"
                :color="dim.color"
                :height="8"
                :border-radius="4"
              />
            </div>
          </NSpace>
        </NCard>

        <!-- 已完成但无评分 -->
        <NCard
          v-if="video.status === 'completed' && !latestScore"
          size="medium"
          bordered
        >
          <NEmpty description="视频已完成推理，但未找到评分记录">
            <template #extra>
              <NButton v-if="canDownloadPdf" type="primary" @click="downloadPdf">
                查看 PDF 报告
              </NButton>
            </template>
          </NEmpty>
        </NCard>

        <!-- PDF 预览（仅 completed 且有 report_path）-->
        <NCard
          v-if="canDownloadPdf"
          title="PDF 报告预览"
          size="medium"
          bordered
        >
          <NSpace vertical :size="8">
            <NSpace align="center">
              <NButton type="primary" size="small" @click="downloadPdf">
                <template #icon>
                  <NIcon :component="DownloadOutline" />
                </template>
                在新窗口打开
              </NButton>
              <NText depth="3" style="font-size: 12px;">
                内嵌预览取决于浏览器支持，建议使用 Chrome / Edge
              </NText>
            </NSpace>
            <iframe
              v-if="pdfUrl"
              :src="pdfUrl"
              style="width: 100%; height: 70vh; border: 1px solid #e0e0e0; border-radius: 4px;"
              :title="`PDF 报告 - ${video.original_filename}`"
            />
          </NSpace>
        </NCard>
      </template>
    </NSpin>
  </NSpace>
</template>
