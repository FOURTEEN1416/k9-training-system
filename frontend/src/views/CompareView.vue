<script setup lang="ts">
import { computed, h, onMounted, ref, shallowRef } from 'vue'
import {
  NCard, NSpace, NText, NSelect, NButton, NIcon, NDatePicker, NEmpty,
  NDataTable, NTag, NSpin, NStatistic, NGrid, NGridItem, NAlert,
  useMessage,
  type DataTableColumns, type SelectOption,
} from 'naive-ui'
import { RefreshOutline, StatsChartOutline, SearchOutline } from '@vicons/ionicons5'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, RadarChart } from 'echarts/charts'
import {
  TitleComponent, TooltipComponent, LegendComponent,
  GridComponent, RadarComponent, DataZoomComponent,
} from 'echarts/components'
import { api, type Dog, type ScoreCompareResponse, type DogScoreStats } from '@/api'

// ECharts 按需注册（vue-echarts 7 推荐方式）
use([
  CanvasRenderer, LineChart, RadarChart,
  TitleComponent, TooltipComponent, LegendComponent,
  GridComponent, RadarComponent, DataZoomComponent,
])

const message = useMessage()

// === 数据 ===
const dogs = ref<Dog[]>([])
const compareResult = ref<ScoreCompareResponse | null>(null)
const loading = ref(false)
const dogsLoading = ref(false)

// === 筛选 ===
const selectedDogIds = ref<number[]>([])
const selectedStandard = ref<string | null>(null)
const dateRange = ref<[number, number] | null>(null)

const dogOptions = computed<SelectOption[]>(() =>
  dogs.value.map(d => ({
    label: `${d.name}${d.breed ? '（' + d.breed + '）' : ''}`,
    value: d.id,
  })),
)

const standardOptions: SelectOption[] = [
  { label: '全部标准', value: '' },
  { label: 'GA-T 公安标准', value: 'GA-T' },
  { label: 'USPCA 美国警犬认证', value: 'USPCA' },
  { label: 'FCI-IGP 国际工作犬', value: 'FCI-IGP' },
  { label: 'CUSTOM 自定义', value: 'CUSTOM' },
]

// === ECharts 配色（每犬一种颜色，最多 10 种）===
const DOG_COLORS = [
  '#2080f0', '#18a058', '#f0a020', '#d03050', '#7c3aed',
  '#0ea5e9', '#10b981', '#ec4899', '#f59e0b', '#6366f1',
]

// === 加载犬只列表 ===
async function loadDogs() {
  dogsLoading.value = true
  try {
    dogs.value = await api.listDogs()
  } catch (e) {
    message.error(`加载犬只列表失败: ${(e as Error).message}`)
  } finally {
    dogsLoading.value = false
  }
}

// === 执行对比查询 ===
async function runCompare() {
  if (selectedDogIds.value.length < 1) {
    message.warning('请至少选择 1 只犬')
    return
  }
  if (selectedDogIds.value.length > 10) {
    message.warning('对比犬只数不能超过 10 只')
    return
  }

  loading.value = true
  try {
    const params: {
      dogIds: number[]
      standard?: string
      dateFrom?: string
      dateTo?: string
    } = { dogIds: selectedDogIds.value }

    if (selectedStandard.value) {
      params.standard = selectedStandard.value
    }
    if (dateRange.value) {
      params.dateFrom = new Date(dateRange.value[0]).toISOString()
      params.dateTo = new Date(dateRange.value[1]).toISOString()
    }

    compareResult.value = await api.compareScores(params)
    // 重建图表配置
    lineOption.value = buildLineOption()
    radarOption.value = buildRadarOption()

    if (compareResult.value.dogs.length === 0) {
      message.info('未找到匹配的犬只')
    } else if (compareResult.value.stats.every(s => s.count === 0)) {
      message.info('所选犬只在当前筛选条件下无评分记录')
    }
  } catch (e) {
    message.error(`对比查询失败: ${(e as Error).message}`)
    compareResult.value = null
  } finally {
    loading.value = false
  }
}

// === 重置筛选 ===
function resetFilter() {
  selectedDogIds.value = []
  selectedStandard.value = null
  dateRange.value = null
  compareResult.value = null
}

// === 趋势斜率展示文本 ===
function formatTrendSlope(slope?: number | null): string {
  if (slope == null) return '—'
  const perDay = slope.toFixed(2)
  if (slope > 0.05) return `↑ +${perDay} 分/天`
  if (slope < -0.05) return `↓ ${perDay} 分/天`
  return `→ ${perDay} 分/天`
}

function trendTagType(slope?: number | null): 'success' | 'error' | 'default' {
  if (slope == null) return 'default'
  if (slope > 0.05) return 'success'
  if (slope < -0.05) return 'error'
  return 'default'
}

// === ECharts: 综合分趋势折线图 ===
const lineOption = shallowRef<Record<string, unknown>>({})

function buildLineOption(): Record<string, unknown> {
  if (!compareResult.value) return {}
  const series = compareResult.value.series

  // 合并所有时间点作为 X 轴
  const allTimes = new Set<string>()
  series.forEach(s => s.points.forEach(p => allTimes.add(p.created_at)))
  const sortedTimes = Array.from(allTimes).sort()

  return {
    title: {
      text: '综合分趋势对比',
      left: 'center',
      textStyle: { fontSize: 14, fontWeight: 'normal' },
    },
    tooltip: {
      trigger: 'axis',
      formatter: (params: Array<{ axisValue: string; seriesName: string; value: number | null }>) => {
        if (!params.length) return ''
        const time = new Date(params[0].axisValue).toLocaleString('zh-CN', { hour12: false })
        const lines = params.map(p =>
          `${p.seriesName}: <strong>${p.value != null ? p.value.toFixed(1) : '—'}</strong>`,
        )
        return [time, ...lines].join('<br/>')
      },
    },
    legend: {
      top: 30,
      data: series.map(s => s.dog.name),
    },
    grid: { top: 80, left: 50, right: 30, bottom: 60 },
    xAxis: {
      type: 'category',
      data: sortedTimes,
      axisLabel: {
        formatter: (val: string) =>
          new Date(val).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }),
      },
    },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100,
      name: '综合分',
    },
    dataZoom: [
      { type: 'inside', start: 0, end: 100 },
      { type: 'slider', start: 0, end: 100, height: 20, bottom: 10 },
    ],
    series: series.map((s, idx) => {
      const color = DOG_COLORS[idx % DOG_COLORS.length]
      // 按时间轴对齐，缺失点为 null
      const data = sortedTimes.map(t => {
        const p = s.points.find(pt => pt.created_at === t)
        return p ? p.overall : null
      })
      return {
        name: s.dog.name,
        type: 'line',
        data,
        smooth: true,
        symbol: 'circle',
        symbolSize: 6,
        lineStyle: { width: 2, color },
        itemStyle: { color },
        connectNulls: true,
      }
    }),
  }
}

// === ECharts: 7 维平均分雷达图 ===
const radarOption = shallowRef<Record<string, unknown>>({})

function buildRadarOption(): Record<string, unknown> {
  if (!compareResult.value) return {}
  const stats = compareResult.value.stats
  const dimLabels = compareResult.value.dimension_labels

  // 找出所有犬只的维度并集
  const allDims = new Set<string>()
  stats.forEach(s => Object.keys(s.avg_dimensions).forEach(d => allDims.add(d)))
  const dims = Array.from(allDims)

  return {
    title: {
      text: '7 维平均分对比',
      left: 'center',
      textStyle: { fontSize: 14, fontWeight: 'normal' },
    },
    tooltip: { trigger: 'item' },
    legend: {
      top: 30,
      data: stats.map(s => s.dog.name),
    },
    radar: {
      indicator: dims.map(d => ({ name: dimLabels[d] || d, max: 100, min: 0 })),
      radius: '65%',
      center: ['50%', '60%'],
    },
    series: [
      {
        type: 'radar',
        data: stats.map((s, idx) => {
          const color = DOG_COLORS[idx % DOG_COLORS.length]
          return {
            name: s.dog.name,
            value: dims.map(d => s.avg_dimensions[d] ?? 0),
            lineStyle: { width: 2, color },
            itemStyle: { color },
            areaStyle: { opacity: 0.15 },
          }
        }),
      },
    ],
  }
}

// === 派生状态 ===
const hasData = computed(() =>
  compareResult.value !== null && compareResult.value.dogs.length > 0,
)

const hasScoreData = computed(() =>
  hasData.value && compareResult.value!.stats.some(s => s.count > 0),
)

// === 对比表 ===
const tableColumns = computed<DataTableColumns<DogScoreStats>>(() => [
  {
    title: '犬只',
    key: 'dog',
    width: 160,
    render: (row, index) =>
      h(NSpace, { align: 'center', size: 8 }, () => [
        h('span', {
          style: `display:inline-block;width:10px;height:10px;border-radius:50%;background:${DOG_COLORS[index % DOG_COLORS.length]};`,
        }),
        h(NText, { strong: true }, () => row.dog.name),
        row.dog.breed
          ? h(NText, { depth: 3, style: 'font-size:12px' }, () => `(${row.dog.breed})`)
          : null,
      ]),
  },
  { title: '评分次数', key: 'count', width: 100, align: 'center' },
  {
    title: '平均分',
    key: 'avg_overall',
    width: 100,
    align: 'center',
    sorter: (a, b) => a.avg_overall - b.avg_overall,
    render: row => row.count > 0 ? row.avg_overall.toFixed(1) : '—',
  },
  {
    title: '最高分',
    key: 'max_overall',
    width: 100,
    align: 'center',
    render: row => row.count > 0 ? row.max_overall.toFixed(1) : '—',
  },
  {
    title: '最低分',
    key: 'min_overall',
    width: 100,
    align: 'center',
    render: row => row.count > 0 ? row.min_overall.toFixed(1) : '—',
  },
  {
    title: '最新分',
    key: 'latest_overall',
    width: 100,
    align: 'center',
    render: row => row.latest_overall != null ? row.latest_overall.toFixed(1) : '—',
  },
  {
    title: '趋势',
    key: 'trend_slope',
    width: 150,
    align: 'center',
    render: row =>
      h(NTag, {
        type: trendTagType(row.trend_slope),
        size: 'small',
        round: true,
      }, () => formatTrendSlope(row.trend_slope)),
  },
])

onMounted(() => {
  loadDogs()
})
</script>

<template>
  <NSpace vertical :size="16">
    <!-- 筛选区 -->
    <NCard title="训练历史对比" size="medium" bordered>
      <NSpace :size="12" align="center" wrap>
        <NSelect
          v-model:value="selectedDogIds"
          :options="dogOptions"
          multiple
          filterable
          max-tag-count="responsive"
          placeholder="选择犬只（1-10 只）"
          style="width: 380px;"
          :loading="dogsLoading"
        />
        <NSelect
          v-model:value="selectedStandard"
          :options="standardOptions"
          placeholder="评分标准"
          style="width: 200px;"
          clearable
        />
        <NDatePicker
          v-model:value="dateRange"
          type="daterange"
          clearable
          placeholder="日期范围（可选）"
          style="width: 280px;"
        />
        <NButton type="primary" :loading="loading" @click="runCompare">
          <template #icon>
            <NIcon :component="SearchOutline" />
          </template>
          对比查询
        </NButton>
        <NButton quaternary @click="resetFilter">
          <template #icon>
            <NIcon :component="RefreshOutline" />
          </template>
          重置
        </NButton>
      </NSpace>
    </NCard>

    <!-- 结果区 -->
    <NSpin :show="loading">
      <template v-if="!hasData">
        <NCard size="medium" bordered>
          <NEmpty description="请选择犬只并点击「对比查询」">
            <template #icon>
              <NIcon :component="StatsChartOutline" :size="48" depth="3" />
            </template>
          </NEmpty>
        </NCard>
      </template>

      <template v-else>
        <!-- 无评分数据提示 -->
        <NAlert
          v-if="!hasScoreData"
          type="info"
          title="未找到评分记录"
          style="margin-bottom: 16px;"
        >
          所选犬只在当前筛选条件下无评分记录。请调整筛选条件或先上传视频并完成评分。
        </NAlert>

        <!-- 统计卡片 -->
        <NGrid :cols="4" :x-gap="12" :y-gap="12" responsive="screen" style="margin-bottom: 16px;">
          <NGridItem
            v-for="(stat, idx) in compareResult!.stats"
            :key="stat.dog.id"
          >
            <NCard size="small" bordered>
              <NSpace vertical :size="4">
                <NSpace align="center" :size="6">
                  <span
                    :style="`display:inline-block;width:12px;height:12px;border-radius:50%;background:${DOG_COLORS[idx % DOG_COLORS.length]};`"
                  />
                  <NText strong>{{ stat.dog.name }}</NText>
                  <NText depth="3" style="font-size: 11px;">
                    {{ stat.dog.breed || '—' }}
                  </NText>
                </NSpace>
                <NSpace :size="16">
                  <NStatistic label="平均" :value="stat.count > 0 ? stat.avg_overall.toFixed(1) : '—'" />
                  <NStatistic label="最新" :value="stat.latest_overall != null ? stat.latest_overall.toFixed(1) : '—'" />
                  <NStatistic label="次数" :value="stat.count" />
                </NSpace>
              </NSpace>
            </NCard>
          </NGridItem>
        </NGrid>

        <!-- 趋势折线图 -->
        <NCard size="medium" bordered style="margin-bottom: 16px;">
          <VChart
            v-if="hasScoreData"
            :option="lineOption"
            autoresize
            style="height: 420px;"
          />
          <NEmpty v-else description="暂无评分数据可用于趋势图" />
        </NCard>

        <!-- 雷达图 + 对比表 -->
        <NGrid :cols="2" :x-gap="16" :y-gap="16" responsive="screen">
          <NGridItem>
            <NCard size="medium" bordered>
              <VChart
                v-if="hasScoreData"
                :option="radarOption"
                autoresize
                style="height: 420px;"
              />
              <NEmpty v-else description="暂无维度数据可用于雷达图" />
            </NCard>
          </NGridItem>
          <NGridItem>
            <NCard size="medium" bordered title="对比明细">
              <NDataTable
                :columns="tableColumns"
                :data="compareResult!.stats"
                :bordered="false"
                :row-key="(row: DogScoreStats) => row.dog.id"
                size="small"
              />
            </NCard>
          </NGridItem>
        </NGrid>
      </template>
    </NSpin>
  </NSpace>
</template>
