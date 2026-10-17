import math
import os
import shutil
import time
from pathlib import Path

from videotrans.configure.config import tr, logger
from videotrans.util.help_ffmpeg import get_audio_time, runffmpeg, create_concat_txt, concat_multi_audio, \
    change_speed_rubberband
from videotrans.util.help_misc import vail_file


class AudioMixin:

    def _back_music(self) -> None:
        if self._exit() or not self.should_dubbing or not vail_file(self.cfg.target_wav) or not vail_file( self.cfg.background_music):
            return
        try:
            self.signal(text=tr("Adding background audio"))
            vtime = get_audio_time(self.cfg.target_wav)
            atime = get_audio_time(self.cfg.background_music)
            if atime < 1:
                return
            bgm_file=self.cfg.cache_folder + f'/bgm_file.wav'
            self._set_volume(self.cfg.background_music, bgm_file)
            self.cfg.background_music = bgm_file
            beishu = math.ceil(vtime / atime)
            if self.cfg.loop_backaudio and beishu > 1 and vtime - 1000 > atime:
                file_list = [self.cfg.background_music for n in range(beishu + 1)]
                concat_txt = self.cfg.cache_folder + f'/{time.time()}.txt'
                create_concat_txt(file_list, concat_txt=concat_txt)
                concat_multi_audio(
                    concat_txt=concat_txt,
                    out=self.cfg.cache_folder + "/bgm_file_extend.wav")
                self.cfg.background_music = self.cfg.cache_folder + "/bgm_file_extend.wav"

            # amix 加 normalize=0 防止默认归一化把人声音量减半；alimiter 防止音量提升后爆音
            cmd = ['-y',
                   '-i', os.path.basename(self.cfg.target_wav),
                   '-i', self.cfg.background_music,
                   '-filter_complex', "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.95:level=0",
                   '-ac', '2',
                   '-c:a', 'pcm_s16le',
                   "lastend.wav"
                   ]
            runffmpeg(cmd, cmd_dir=self.cfg.cache_folder)
            self.cfg.target_wav = self.cfg.cache_folder + f"/lastend.wav"
        except Exception as e:
            logger.exception(f'添加背景音乐失败,静默跳过 {e}', exc_info=True)

    def _separate(self) -> None:
        if self._exit() or not self.cfg.embed_bgm or not vail_file(self.cfg.instrument) or not vail_file(self.cfg.target_wav):
            return
        try:
            self.signal(text=tr("Re-embedded background sounds"))
            vtime = get_audio_time(self.cfg.target_wav)
            atime = get_audio_time(self.cfg.instrument)
            if atime < 1:
                return
            beishu = math.ceil(vtime / atime)

            instrument_file = self.cfg.instrument
            logger.debug(f'合并背景音 {beishu=},{atime=},{vtime=}')
            if atime + 1000 < vtime:
                if int(self.cfg.loop_backaudio) == 1:
                    file_list = [instrument_file for n in range(beishu + 1)]
                    concat_txt = self.cfg.cache_folder + f'/{time.time()}.txt'
                    create_concat_txt(file_list, concat_txt=concat_txt)
                    concat_multi_audio(concat_txt=concat_txt,
                                             out=self.cfg.cache_folder + "/instrument-concat.wav")
                else:
                    change_speed_rubberband(instrument_file, self.cfg.cache_folder + f"/instrument-concat.wav",
                                                  vtime)
                instrument_file = self.cfg.cache_folder + f"/instrument-concat.wav"

            tmp_out_wav = Path(self.cfg.cache_folder + f'/{time.time()}-1.wav').as_posix()
            tmp_volume = Path(self.cfg.cache_folder + f'/{time.time()}.wav').as_posix()

            self._set_volume(instrument_file, tmp_volume)
            effects = self._effects_without_speech()
            mix_inputs = ['-i', os.path.basename(self.cfg.target_wav), '-i', os.path.basename(tmp_volume)]
            if effects:
                mix_inputs.extend(['-i', os.path.basename(effects)])
                mix = "[0:a][1:a][2:a]amix=inputs=3:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.95:level=0"
                logger.debug(f'保留对白之间的原声效果: {effects}')
            else:
                mix = "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,alimiter=limit=0.95:level=0"
            runffmpeg(['-y', *mix_inputs,
                             '-filter_complex',
                             # amix 加 normalize=0 防止默认归一化把人声音量减半；alimiter 防止音量提升后爆音
                             mix, '-ac', '2', "-b:a", "128k",
                             '-c:a', 'pcm_s16le', os.path.basename(tmp_out_wav)], cmd_dir=self.cfg.cache_folder)
            shutil.copy2(tmp_out_wav, self.cfg.target_wav)
        except Exception as e:
            logger.exception(f'重新嵌入分离的背景音失败 {e}', exc_info=True)
    
    def _effects_without_speech(self):
        """Vocal-stem sounds that are not dialogue, such as R2-D2."""
        vocal = getattr(self.cfg, "vocal", None)
        if not vocal or not vail_file(vocal):
            return None
        lines = list(self.source_srt_list or []) + list(self.target_srt_list or [])
        from videotrans.task._keep_effects import speech_mute_filter
        audio_filter = speech_mute_filter(lines)
        if not audio_filter:
            return None
        effects = self.cfg.cache_folder + "/effects-nospeech.wav"
        runffmpeg([
            "-y",
            "-i",
            vocal,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            "-filter:a",
            audio_filter,
            effects,
        ], force_cpu=True)
        return effects if vail_file(effects) else None

    def _set_volume(self,input_file,output_file):
        runffmpeg([
            "-y",
            "-i",
            input_file,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-c:a",
            "pcm_s16le",
            "-filter:a", 
            f"volume={self.cfg.backaudio_volume}",
        
            output_file
        ], force_cpu=True)
