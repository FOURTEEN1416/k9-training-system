<script setup lang="ts">
import { NCard, NSpace, NText, NUpload, NButton, NIcon, useMessage, type UploadFileInfo } from 'naive-ui'
import { CloudUploadOutline } from '@vicons/ionicons5'
import { ref } from 'vue'

const message = useMessage()
const fileList = ref<UploadFileInfo[]>([])

function handleChange(options: { fileList: UploadFileInfo[] }) {
  fileList.value = options.fileList
}

function handleUploadClick() {
  message.info('视频上传将在 Phase 1 实现')
}
</script>

<template>
  <NSpace vertical :size="16">
    <NCard title="视频上传" size="medium" bordered>
      <template #header-extra>
        <NText depth="3" style="font-size: 12px;">Phase 1 实现</NText>
      </template>
      <NSpace vertical :size="12">
        <NText>选择训练视频文件（MP4 / AVI / MOV）</NText>
        <NUpload
          v-model:file-list="fileList"
          :max="10"
          multiple
          :default-upload="false"
          @change="handleChange"
        >
          <NButton type="primary" size="large" @click="handleUploadClick">
            <template #icon>
              <NIcon :component="CloudUploadOutline" />
            </template>
            选择视频文件
          </NButton>
        </NUpload>
      </NSpace>
    </NCard>

    <NCard title="训练会话关联" size="medium" bordered>
      <NText depth="3">犬只 / 训导员 / 训练日期 / 评分标准将在此选择</NText>
    </NCard>
  </NSpace>
</template>
