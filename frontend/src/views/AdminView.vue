<script setup lang="ts">
import { computed, h, onMounted, ref } from 'vue'
import {
  NCard, NSpace, NText, NTag, NButton, NIcon, NSpin, NDescriptions,
  NDescriptionsItem, NDataTable, NModal, NForm, NFormItem, NInput, NSelect,
  NSwitch, NInputNumber, NGrid, NGridItem, NAlert, NStatistic,
  NRadioGroup, NRadio, useMessage,
  type DataTableColumns, type SelectOption, type FormInst, type FormRules,
} from 'naive-ui'
import {
  RefreshOutline, AddOutline, SaveOutline, PlayOutline,
  PencilOutline, TrashOutline,
} from '@vicons/ionicons5'
import {
  api, type MlModel, type ScoringConfig, type ScoringResult,
  type Scene, type Dog,
} from '@/api'
import { useAppStore } from '@/stores/app'

const message = useMessage()
const appStore = useAppStore()

// ============================================================
// Tab 切换
// ============================================================
type TabKey = 'models' | 'scoring' | 'dogs' | 'system'
const activeTab = ref<TabKey>('models')

const tabs: Array<{ key: TabKey; label: string }> = [
  { key: 'models', label: '模型管理' },
  { key: 'scoring', label: '评分卡配置' },
  { key: 'dogs', label: '犬只档案' },
  { key: 'system', label: '系统信息' },
]

// ============================================================
// 模型管理
// ============================================================
const models = ref<MlModel[]>([])
const loadingModels = ref(false)

const modelTypeFilter = ref<string | null>(null)
const modelTypeOptions: SelectOption[] = [
  { label: '全部类型', value: '' },
  { label: '姿态检测 (pose)', value: 'pose' },
  { label: '行为识别 (behavior)', value: 'behavior' },
  { label: '评分引擎 (scoring)', value: 'scoring' },
]

const filteredModels = computed(() =>
  modelTypeFilter.value
    ? models.value.filter(m => m.type === modelTypeFilter.value)
    : models.value,
)

async function loadModels() {
  loadingModels.value = true
  try {
    models.value = await api.listModels()
  } catch (e) {
    message.error(`加载模型列表失败: ${(e as Error).message}`)
  } finally {
    loadingModels.value = false
  }
}

async function activateModel(model: MlModel) {
  try {
    await api.activateModel(model.id)
    message.success(`已激活: ${model.name} ${model.version}`)
    await loadModels()
  } catch (e) {
    message.error(`激活失败: ${(e as Error).message}`)
  }
}

// 注册新模型
const showRegisterModal = ref(false)
const registerFormRef = ref<FormInst | null>(null)
const registerForm = ref({
  name: '',
  version: '',
  type: 'pose',
  framework: 'pytorch',
  storage_path: '',
  is_active: false,
  description: '',
  metrics_text: '',
})

const registerRules: FormRules = {
  name: { required: true, message: '请输入模型名称', trigger: 'blur' },
  version: { required: true, message: '请输入版本号', trigger: 'blur' },
  storage_path: { required: true, message: '请输入模型存储路径', trigger: 'blur' },
}

const typeOptions: SelectOption[] = [
  { label: '姿态检测 (pose)', value: 'pose' },
  { label: '行为识别 (behavior)', value: 'behavior' },
  { label: '评分引擎 (scoring)', value: 'scoring' },
]

const frameworkOptions: SelectOption[] = [
  { label: 'PyTorch (.pt)', value: 'pytorch' },
  { label: 'ONNX Runtime (.onnx)', value: 'onnx' },
  { label: 'TensorRT (.engine)', value: 'tensorrt' },
]

function openRegisterModal() {
  registerForm.value = {
    name: '', version: '', type: 'pose', framework: 'pytorch',
    storage_path: '', is_active: false, description: '', metrics_text: '',
  }
  showRegisterModal.value = true
}

async function submitRegister() {
  try {
    await registerFormRef.value?.validate()
  } catch {
    return
  }

  let metrics: Record<string, unknown> | undefined = undefined
  if (registerForm.value.metrics_text.trim()) {
    try {
      metrics = JSON.parse(registerForm.value.metrics_text)
    } catch {
      message.error('metrics 不是合法 JSON')
      return
    }
  }

  try {
    await api.registerModel({
      name: registerForm.value.name,
      version: registerForm.value.version,
      type: registerForm.value.type,
      framework: registerForm.value.framework,
      storage_path: registerForm.value.storage_path,
      is_active: registerForm.value.is_active,
      description: registerForm.value.description || undefined,
      metrics: metrics as Record<string, unknown> | undefined,
    } as any)
    message.success('模型注册成功')
    showRegisterModal.value = false
    await loadModels()
  } catch (e) {
    message.error(`注册失败: ${(e as Error).message}`)
  }
}

const modelColumns: DataTableColumns<MlModel> = [
  { title: 'ID', key: 'id', width: 60 },
  { title: '名称', key: 'name', width: 160 },
  { title: '版本', key: 'version', width: 100 },
  { title: '类型', key: 'type', width: 100 },
  { title: '框架', key: 'framework', width: 110 },
  { title: '路径', key: 'storage_path', ellipsis: { tooltip: true } },
  {
    title: '状态',
    key: 'is_active',
    width: 90,
    render: row =>
      h(NTag, {
        type: row.is_active ? 'success' : 'default',
        size: 'small',
        round: true,
      }, () => row.is_active ? '已激活' : '未激活'),
  },
  {
    title: '操作',
    key: 'actions',
    width: 110,
    render: row =>
      h(NButton, {
        size: 'tiny',
        type: row.is_active ? 'default' : 'primary',
        disabled: row.is_active,
        quaternary: !row.is_active,
        onClick: () => activateModel(row),
      }, () => row.is_active ? '当前' : '激活'),
  },
]

// ============================================================
// 评分卡配置
// ============================================================
const scoringConfigs = ref<ScoringConfig[]>([])
const currentScene = ref<Scene>('obedience_trial')
const yamlContent = ref('')
const yamlLoading = ref(false)
const yamlSaving = ref(false)

async function loadAllScoringConfigs() {
  yamlLoading.value = true
  try {
    scoringConfigs.value = await api.listScoringConfigs()
    await loadCurrentConfig()
  } catch (e) {
    message.error(`加载评分卡失败: ${(e as Error).message}`)
  } finally {
    yamlLoading.value = false
  }
}

async function loadCurrentConfig() {
  yamlLoading.value = true
  try {
    const cfg = await api.getScoringConfig(currentScene.value)
    yamlContent.value = cfg.content
  } catch (e) {
    message.error(`加载评分卡失败: ${(e as Error).message}`)
  } finally {
    yamlLoading.value = false
  }
}

async function saveScoringConfig() {
  yamlSaving.value = true
  try {
    await api.updateScoringConfig(currentScene.value, yamlContent.value)
    message.success('评分卡已保存（触发热加载）')
    await loadAllScoringConfigs()
  } catch (e) {
    message.error(`保存失败: ${(e as Error).message}`)
  } finally {
    yamlSaving.value = false
  }
}

function switchScene(scene: Scene) {
  currentScene.value = scene
  loadCurrentConfig()
}

// 试评分
const testSignalsText = ref('{\n  "behavior_count": 3,\n  "duration_sec": 45,\n  "is_correct": true\n}')
const testResult = ref<ScoringResult | null>(null)
const testing = ref(false)

async function runEvaluate() {
  let signals: Record<string, unknown>
  try {
    signals = JSON.parse(testSignalsText.value)
  } catch (e) {
    message.error(`signals 不是合法 JSON: ${(e as Error).message}`)
    return
  }

  testing.value = true
  testResult.value = null
  try {
    testResult.value = await api.evaluateScore(currentScene.value, signals as any)
    message.success(`评分完成: ${testResult.value.total_score} 分`)
  } catch (e) {
    message.error(`评分失败: ${(e as Error).message}`)
  } finally {
    testing.value = false
  }
}

const verdictTagType: Record<string, 'success' | 'warning' | 'error'> = {
  pass: 'success',
  borderline: 'warning',
  fail: 'error',
}

// ============================================================
// 犬只档案
// ============================================================
const dogs = ref<Dog[]>([])
const loadingDogs = ref(false)

const showDogModal = ref(false)
const dogFormRef = ref<FormInst | null>(null)
const editingDogId = ref<number | null>(null)
const dogForm = ref({
  name: '',
  breed: '',
  gender: 'male',
  chip_id: '',
  handler_id: null as number | null,
  training_stage: 'P0',
})

const dogRules: FormRules = {
  name: { required: true, message: '请输入犬只名称', trigger: 'blur' },
}

const genderOptions: SelectOption[] = [
  { label: '公', value: 'male' },
  { label: '母', value: 'female' },
]

const stageOptions: SelectOption[] = [
  { label: 'P0 - 幼犬', value: 'P0' },
  { label: 'P1 - 训练中', value: 'P1' },
  { label: 'P2 - 高级', value: 'P2' },
  { label: 'P3 - 服役', value: 'P3' },
]

async function loadDogs() {
  loadingDogs.value = true
  try {
    dogs.value = await api.listDogs()
  } catch (e) {
    message.error(`加载犬只列表失败: ${(e as Error).message}`)
  } finally {
    loadingDogs.value = false
  }
}

function openCreateDog() {
  editingDogId.value = null
  dogForm.value = {
    name: '', breed: '', gender: 'male', chip_id: '',
    handler_id: null, training_stage: 'P0',
  }
  showDogModal.value = true
}

function openEditDog(dog: Dog) {
  editingDogId.value = dog.id
  dogForm.value = {
    name: dog.name,
    breed: dog.breed || '',
    gender: dog.gender || 'male',
    chip_id: dog.chip_id || '',
    handler_id: dog.handler_id ?? null,
    training_stage: dog.training_stage,
  }
  showDogModal.value = true
}

async function submitDog() {
  try {
    await dogFormRef.value?.validate()
  } catch {
    return
  }

  const payload: Partial<Dog> = {
    name: dogForm.value.name,
    breed: dogForm.value.breed || null,
    gender: dogForm.value.gender,
    chip_id: dogForm.value.chip_id || null,
    handler_id: dogForm.value.handler_id,
    training_stage: dogForm.value.training_stage,
  }

  try {
    if (editingDogId.value == null) {
      await api.createDog(payload)
      message.success('犬只档案已创建')
    } else {
      await api.updateDog(editingDogId.value, payload)
      message.success('犬只档案已更新')
    }
    showDogModal.value = false
    await loadDogs()
  } catch (e) {
    message.error(`保存失败: ${(e as Error).message}`)
  }
}

async function deleteDog(dog: Dog) {
  if (!confirm(`确定删除犬只「${dog.name}」？此操作不可恢复。`)) return
  try {
    await api.deleteDog(dog.id)
    message.success('已删除')
    await loadDogs()
  } catch (e) {
    message.error(`删除失败: ${(e as Error).message}`)
  }
}

const dogColumns: DataTableColumns<Dog> = [
  { title: 'ID', key: 'id', width: 60 },
  { title: '名称', key: 'name', width: 120 },
  { title: '品种', key: 'breed', width: 120 },
  {
    title: '性别',
    key: 'gender',
    width: 70,
    render: row => row.gender === 'male' ? '公' : row.gender === 'female' ? '母' : '—',
  },
  { title: '芯片 ID', key: 'chip_id', width: 140 },
  { title: '训导员 ID', key: 'handler_id', width: 90 },
  { title: '阶段', key: 'training_stage', width: 80 },
  {
    title: '操作',
    key: 'actions',
    width: 140,
    render: row =>
      h(NSpace, { size: 4 }, () => [
        h(NButton, {
          size: 'tiny', quaternary: true, type: 'primary',
          onClick: () => openEditDog(row),
        }, {
          default: () => '编辑',
          icon: () => h(NIcon, null, () => h(PencilOutline)),
        }),
        h(NButton, {
          size: 'tiny', quaternary: true, type: 'error',
          onClick: () => deleteDog(row),
        }, {
          default: () => '删除',
          icon: () => h(NIcon, null, () => h(TrashOutline)),
        }),
      ]),
  },
]

// ============================================================
// 系统信息
// ============================================================

// ============================================================
// 初始化
// ============================================================
onMounted(async () => {
  await Promise.all([
    loadModels(),
    loadAllScoringConfigs(),
    loadDogs(),
  ])
})
</script>

<template>
  <NSpace vertical :size="16">
    <!-- Tab 切换 -->
    <NCard size="small" bordered>
      <NSpace :size="4">
        <NButton
          v-for="tab in tabs"
          :key="tab.key"
          :type="activeTab === tab.key ? 'primary' : 'default'"
          :ghost="activeTab === tab.key"
          size="small"
          @click="activeTab = tab.key"
        >
          {{ tab.label }}
        </NButton>
      </NSpace>
    </NCard>

    <!-- 模型管理 -->
    <template v-if="activeTab === 'models'">
      <NCard title="模型版本管理" size="medium" bordered>
        <template #header-extra>
          <NSpace :size="8">
            <NSelect
              v-model:value="modelTypeFilter"
              :options="modelTypeOptions"
              size="small"
              style="width: 180px;"
              clearable
              placeholder="按类型筛选"
            />
            <NButton size="small" quaternary @click="loadModels">
              <template #icon>
                <NIcon :component="RefreshOutline" />
              </template>
              刷新
            </NButton>
            <NButton size="small" type="primary" @click="openRegisterModal">
              <template #icon>
                <NIcon :component="AddOutline" />
              </template>
              注册新模型
            </NButton>
          </NSpace>
        </template>

        <NSpin :show="loadingModels">
          <NDataTable
            :columns="modelColumns"
            :data="filteredModels"
            :bordered="false"
            :pagination="{ pageSize: 10 }"
            :row-key="(row: MlModel) => row.id"
          />
        </NSpin>
      </NCard>
    </template>

    <!-- 评分卡配置 -->
    <template v-if="activeTab === 'scoring'">
      <NCard title="评分卡 YAML 编辑器" size="medium" bordered>
        <template #header-extra>
          <NSpace :size="8">
            <NRadioGroup v-model:value="currentScene" @update:value="switchScene">
              <NRadio value="obedience_trial">科目测评</NRadio>
              <NRadio value="puppy_selection">幼犬选育</NRadio>
            </NRadioGroup>
            <NButton size="small" quaternary @click="loadCurrentConfig">
              <template #icon>
                <NIcon :component="RefreshOutline" />
              </template>
              重载
            </NButton>
            <NButton
              size="small"
              type="primary"
              :loading="yamlSaving"
              @click="saveScoringConfig"
            >
              <template #icon>
                <NIcon :component="SaveOutline" />
              </template>
              保存（触发热加载）
            </NButton>
          </NSpace>
        </template>

        <NSpin :show="yamlLoading">
          <NInput
            v-model:value="yamlContent"
            type="textarea"
            :rows="20"
            placeholder="评分卡 YAML 内容"
            style="font-family: 'Consolas', 'Courier New', monospace; font-size: 13px;"
          />
        </NSpin>

        <NSpace align="center" style="margin-top: 12px;">
          <NText depth="3" style="font-size: 12px;">
            提示: 保存前请确保 YAML 语法正确，Schema 校验失败会拒绝保存。
          </NText>
        </NSpace>
      </NCard>

      <!-- 试评分 -->
      <NCard title="试评分（实时调试）" size="medium" bordered>
        <NGrid :cols="2" :x-gap="16" :y-gap="16">
          <NGridItem>
            <NSpace vertical :size="8">
              <NText strong>信号输入 (JSON)</NText>
              <NInput
                v-model:value="testSignalsText"
                type="textarea"
                :rows="12"
                placeholder='{"behavior_count": 3, "duration_sec": 45}'
                style="font-family: 'Consolas', 'Courier New', monospace; font-size: 13px;"
              />
              <NSpace>
                <NButton
                  type="primary"
                  :loading="testing"
                  @click="runEvaluate"
                >
                  <template #icon>
                    <NIcon :component="PlayOutline" />
                  </template>
                  执行评分
                </NButton>
                <NText depth="3" style="font-size: 12px; line-height: 32px;">
                  场景: {{ currentScene }}
                </NText>
              </NSpace>
            </NSpace>
          </NGridItem>
          <NGridItem>
            <NSpace vertical :size="12">
              <NText strong>评分结果</NText>
              <div v-if="testResult">
                <NGrid :cols="3" :x-gap="12" :y-gap="12">
                  <NGridItem>
                    <NStatistic label="总分" :value="testResult.total_score" />
                  </NGridItem>
                  <NGridItem>
                    <NStatistic label="判定">
                      <template #default>
                        <NTag
                          :type="verdictTagType[testResult.verdict] || 'default'"
                          size="large"
                          round
                        >
                          {{ testResult.verdict }}
                        </NTag>
                      </template>
                    </NStatistic>
                  </NGridItem>
                  <NGridItem>
                    <NStatistic label="是否合格">
                      <template #default>
                        <NTag :type="testResult.passed ? 'success' : 'error'" size="large" round>
                          {{ testResult.passed ? '合格' : '不合格' }}
                        </NTag>
                      </template>
                    </NStatistic>
                  </NGridItem>
                </NGrid>

                <NText strong style="display: block; margin: 16px 0 8px;">各维度得分</NText>
                <NSpace vertical :size="4">
                  <NSpace
                    v-for="(score, key) in testResult.dimension_scores"
                    :key="key"
                    align="center"
                    justify="space-between"
                    style="padding: 4px 8px; background: #f9f9f9; border-radius: 4px;"
                  >
                    <NText>{{ testResult.dimension_labels[key] || key }}</NText>
                    <NText strong>{{ score }}</NText>
                  </NSpace>
                </NSpace>

                <NText strong style="display: block; margin: 16px 0 8px;">评分说明</NText>
                <NSpace vertical :size="2">
                  <NText
                    v-for="(line, i) in testResult.explanation"
                    :key="i"
                    depth="3"
                    style="font-size: 13px;"
                  >
                    · {{ line }}
                  </NText>
                </NSpace>

                <NText depth="3" style="display: block; margin-top: 12px; font-size: 12px;">
                  评分卡: {{ testResult.card_name }} v{{ testResult.card_version }}
                </NText>
              </div>
              <NAlert v-else type="default" :bordered="false">
                点击「执行评分」查看结果
              </NAlert>
            </NSpace>
          </NGridItem>
        </NGrid>
      </NCard>
    </template>

    <!-- 犬只档案 -->
    <template v-if="activeTab === 'dogs'">
      <NCard title="犬只档案管理" size="medium" bordered>
        <template #header-extra>
          <NSpace :size="8">
            <NButton size="small" quaternary @click="loadDogs">
              <template #icon>
                <NIcon :component="RefreshOutline" />
              </template>
              刷新
            </NButton>
            <NButton size="small" type="primary" @click="openCreateDog">
              <template #icon>
                <NIcon :component="AddOutline" />
              </template>
              新增犬只
            </NButton>
          </NSpace>
        </template>

        <NSpin :show="loadingDogs">
          <NDataTable
            :columns="dogColumns"
            :data="dogs"
            :bordered="false"
            :pagination="{ pageSize: 20 }"
            :row-key="(row: Dog) => row.id"
          />
        </NSpin>
      </NCard>
    </template>

    <!-- 系统信息 -->
    <template v-if="activeTab === 'system'">
      <NCard title="系统信息" size="medium" bordered>
        <NDescriptions :column="2" label-placement="left" bordered>
          <NDescriptionsItem label="系统名称">工作犬训练机器视觉识别系统</NDescriptionsItem>
          <NDescriptionsItem label="当前阶段">
            <NTag type="info" size="small">Phase 1 MVP</NTag>
          </NDescriptionsItem>
          <NDescriptionsItem label="后端状态">
            <NTag :type="appStore.backendOnline ? 'success' : 'error'" size="small">
              {{ appStore.backendOnline ? '在线' : '离线' }}
            </NTag>
          </NDescriptionsItem>
          <NDescriptionsItem label="后端版本">
            {{ appStore.health?.version || '—' }}
          </NDescriptionsItem>
          <NDescriptionsItem label="运行环境">
            {{ appStore.health?.environment || '—' }}
          </NDescriptionsItem>
          <NDescriptionsItem label="API 文档">
            <a href="/docs" target="_blank">Swagger UI</a>
          </NDescriptionsItem>
        </NDescriptions>
      </NCard>

      <NCard title="关于" size="medium" bordered>
        <NSpace vertical :size="8">
          <NText>
            工作犬训练机器视觉识别系统 · 本地部署 · 支持 GA-T / USPCA / FCI-IGP 三大标准
          </NText>
          <NText depth="3" style="font-size: 13px;">
            双场景: 幼犬选育（食物/玩具/胆量 3 维）+ 科目测评（坐/卧/立/吠叫等 8 类行为）
          </NText>
        </NSpace>
      </NCard>
    </template>

    <!-- 注册模型 Modal -->
    <NModal
      v-model:show="showRegisterModal"
      preset="card"
      title="注册新模型"
      style="width: 600px;"
      :mask-closable="false"
    >
      <NForm
        ref="registerFormRef"
        :model="registerForm"
        :rules="registerRules"
        label-placement="left"
        label-width="100"
      >
        <NFormItem label="模型名称" path="name">
          <NInput v-model:value="registerForm.name" placeholder="如 yolo26n-pose" />
        </NFormItem>
        <NFormItem label="版本" path="version">
          <NInput v-model:value="registerForm.version" placeholder="如 v1.0.0" />
        </NFormItem>
        <NFormItem label="类型" path="type">
          <NSelect v-model:value="registerForm.type" :options="typeOptions" />
        </NFormItem>
        <NFormItem label="框架" path="framework">
          <NSelect v-model:value="registerForm.framework" :options="frameworkOptions" />
        </NFormItem>
        <NFormItem label="存储路径" path="storage_path">
          <NInput
            v-model:value="registerForm.storage_path"
            placeholder="如 backend/ml/pose/weights/yolo26n-pose.pt"
          />
        </NFormItem>
        <NFormItem label="是否激活">
          <NSwitch v-model:value="registerForm.is_active">
            <template #checked>激活（替换同类型当前模型）</template>
            <template #unchecked>仅注册</template>
          </NSwitch>
        </NFormItem>
        <NFormItem label="描述">
          <NInput
            v-model:value="registerForm.description"
            type="textarea"
            :rows="2"
            placeholder="可选"
          />
        </NFormItem>
        <NFormItem label="metrics JSON">
          <NInput
            v-model:value="registerForm.metrics_text"
            type="textarea"
            :rows="3"
            placeholder='{"mAP50": 0.85, "mAP50-95": 0.62}'
          />
        </NFormItem>
        <NSpace justify="end">
          <NButton @click="showRegisterModal = false">取消</NButton>
          <NButton type="primary" @click="submitRegister">注册</NButton>
        </NSpace>
      </NForm>
    </NModal>

    <!-- 犬只档案 Modal -->
    <NModal
      v-model:show="showDogModal"
      preset="card"
      :title="editingDogId == null ? '新增犬只档案' : `编辑犬只 #${editingDogId}`"
      style="width: 560px;"
      :mask-closable="false"
    >
      <NForm
        ref="dogFormRef"
        :model="dogForm"
        :rules="dogRules"
        label-placement="left"
        label-width="100"
      >
        <NFormItem label="名称" path="name">
          <NInput v-model:value="dogForm.name" placeholder="如 黑狼" />
        </NFormItem>
        <NFormItem label="品种">
          <NInput v-model:value="dogForm.breed" placeholder="如 比利时马犬" />
        </NFormItem>
        <NFormItem label="性别">
          <NSelect v-model:value="dogForm.gender" :options="genderOptions" />
        </NFormItem>
        <NFormItem label="芯片 ID">
          <NInput v-model:value="dogForm.chip_id" placeholder="可选" />
        </NFormItem>
        <NFormItem label="训导员 ID">
          <NInputNumber
            v-model:value="dogForm.handler_id"
            placeholder="可选"
            style="width: 100%;"
            :show-button="false"
          />
        </NFormItem>
        <NFormItem label="训练阶段">
          <NSelect v-model:value="dogForm.training_stage" :options="stageOptions" />
        </NFormItem>
        <NSpace justify="end">
          <NButton @click="showDogModal = false">取消</NButton>
          <NButton type="primary" @click="submitDog">
            {{ editingDogId == null ? '创建' : '保存' }}
          </NButton>
        </NSpace>
      </NForm>
    </NModal>
  </NSpace>
</template>
