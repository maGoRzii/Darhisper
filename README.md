# Darhisper 🦅🎙️

> **Tu asistente de voz y herramientas de transcripción definitivo para macOS. Transcripción instantánea, local y privada.**

![macOS](https://img.shields.io/badge/macOS-Apple_Silicon-white?logo=apple&logoColor=black) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![MLX](https://img.shields.io/badge/Powered_by-Apple_MLX-yellow)

**Darhisper** es una suite de productividad diseñada exclusivamente para **macOS (Apple Silicon)**. Combina un asistente residente en la barra de menú para dictado instantáneo en cualquier app con un potente **panel de control** para transcribir archivos de audio de larga duración, todo utilizando la potencia del motor neuronal de tu Mac o la nube.

---

## ✨ Características Principales

*   **⚡️ Velocidad Ultrarrapida (Local)**: Utiliza `mlx-whisper` y `parakeet-tdt` optimizados para chips Apple Silicon, ofreciendo transcripciones en tiempo real sin internet.
*   **📁 Transcripción de Archivos**: Sube tus grabaciones (mp3, wav, m4a...) y conviértelas a texto. Soporta archivos de larga duración (reuniones, clases) mediante procesamiento inteligente por bloques. Guarda automáticamente en `.txt`.
*   **🖥️ Panel de Control Visual**: Una interfaz nativa de macOS moderna y elegante para gestionar tus transcripciones, configurar modelos y ajustar preferencias.
*   **☁️ Potencia en la Nube (Opcional)**: Integración nativa con **Google Gemini 3.0 Flash** para un entendimiento superior del contexto y formateo avanzado.
*   **🎨 Diseño Elegante**: Feedback visual moderno con una interfaz de ondas de voz animadas que flotan sobre tu pantalla mientras dictas.
*   **⌨️ Escribe Donde Sea**: Funciona globalmente. Simplemente coloca el cursor, mantén presionado tu atajo y habla. El texto se escribe mágicamente en la aplicación activa.
*   **⚙️ Totalmente Configurable**:
    *   Cambia de modelos de IA al vuelo.
    *   Graba tus propios atajos de teclado personalizados.
    *   Gestiona tus claves de API de forma segura.

---

## 🖥️ Requisitos del Sistema

Para garantizar el máximo rendimiento, Darhisper tiene requisitos específicos:

*   **Hardware**: Mac con chip **Apple Silicon** (M1, M1 Pro/Max/Ultra, M2, M3, etc.).
    *   *Nota: No es compatible con Macs basados en Intel debido a la dependencia de MLX.*
*   **Sistema Operativo**: macOS 12.0 (Monterey) o superior.
*   **Software Adicional**: `ffmpeg` es necesario para la conversión de archivos de audio.
    *   Instalar con homebrew: `brew install ffmpeg`
*   **Permisos**: Requiere acceso a **Micrófono** y **Accesibilidad** (para la inserción de texto).

---

## 🚀 Instalación y Uso

### Opción A: Para Usuarios (Aplicación Compilada)

1.  **Descarga**: Obtén la última versión de `Darhisper.app` (desde la carpeta `dist` si lo has compilado tú mismo).
2.  **Instala**: Arrastra la app a tu carpeta de **Aplicaciones**.
3.  **Primer Lanzamiento**:
    *   Al abrir la app, verás un icono 🎙️ en la barra de menú.
    *   **Importante**: Si macOS indica que la app "está dañada" o "no se puede abrir", ejecuta este comando en la Terminal para firmarla localmente:
        ```bash
        xattr -cr /Applications/Darhisper.app
        ```
4.  **Concede Permisos**: La primera vez que intentes usarla, macOS te pedirá permisos. Acepta:
    *   🎤 Micrófono.
    *   ⌨️ Accesibilidad/Eventos del sistema (para pegar el texto).

### Opción B: Para Desarrolladores (Código Fuente)

Si prefieres ejecutarlo desde el código o contribuir:

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/maGoRzii/Darhisper.git
    cd Darhisper
    ```

2.  **Configurar entorno**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *Es posible que necesites instalar `portaudio` para el audio:* `brew install portaudio`

3.  **Ejecutar**:
    ```bash
    ./start.sh
    ```

---

## 📖 Guía de Uso

### 1. Panel de Control (Dashboard)
Para acceder a todas las funciones, haz clic en el icono 🎙️ de la barra de menú y selecciona **"Abrir Darhisper"**. Desde aquí puedes:
*   Configurar modelos y atajos.
*   Gestionar claves de API.
*   **Transcribir archivos de audio**.

### 2. Dictado Instantáneo (Modo Barra de Menú)
Ideal para correos, notas rápidas y chats.
1.  Haz clic donde quieras escribir (Slack, Notion, VS Code, etc.).
2.  **Mantén presionado** el atajo de teclado (Por defecto: `F5` o `Opción Derecha`).
3.  Espera el **Beep** y habla cuando veas la **onda de voz** en pantalla.
4.  Suelta la tecla al terminar. El texto aparecerá automáticamente.

### 3. Transcripción de Archivos
Ideal para reuniones grabadas, clases o notas de voz largas.
1.  Abre el **Panel de Control** ("Abrir Darhisper").
2.  En la sección "TRANSCRIPCIÓN DE ARCHIVO", haz clic en **"📁 Elegir Archivo..."**.
3.  Selecciona tu audio (`mp3`, `wav`, `m4a`, `ogg`...).
4.  Haz clic en **"🚀 COMENZAR TRANSCRIPCIÓN"**.
5.  El sistema procesará el audio (dividiéndolo en bloques si es necesario).
6.  Al finalizar:
    *   El texto aparecerá en el cuadro inferior.
    *   Se guardará automáticamente un archivo `.txt` junto al audio original.
    *   Puedes copiarlo al portapapeles con el botón "📋 Copiar".

### Configuración Avanzada
Todas las configuraciones se gestionan desde el **Panel de Control** ("Abrir Darhisper").

#### 🧠 Selección de Modelos
*   **Micrófono (Tiempo Real)**:
    *   *Whisper (Tiny/Base/Small)*: Extremadamente rápidos.
    *   *Large-v3-Turbo / Q4*: Balance perfecto entre precisión y velocidad.
    *   *Parakeet TDT*: Modelo RNN ultra-rápido (0.6B).
*   **Archivos**:
    *   *Gemini Flash*: Máxima precisión y formateo inteligente.
    *   *Parakeet TDT*: Transcripción local a velocidad extrema.

#### 🎭 Selección de Modos (Smart Prompts)
*(Disponible con modelos Gemini)*
Personaliza el estilo de la transcripción:
*   **Transcripción Literal**: Texto exacto, letra por letra.
*   **Lista de Tareas (To-Do)**: Convierte voz en checklist.
*   **Email Profesional**: Redacta correos formales.
*   **Modo Excel/Datos**: Formato tabular para hojas de cálculo.

#### ⌨️ Atajos
*   Elige entre `F5`, `Cmd+Opt+R`, o `Right Option`.
*   Configura tu propio atajo personalizado.

#### 🔐 API Keys (Gemini)
*   Configura tu clave de Google Gemini directamente en el panel para habilitar los modelos en la nube.
*   La clave se guarda de forma segura en tu equipo.

---

## ❓ Solución de Problemas

| Problema | Solución |
| :--- | :--- |
| **No escribe nada** | Verifica que has dado permisos de **Accesibilidad** en *Preferencias del Sistema -> Privacidad y Seguridad*. |
| **Error al iniciar** | Asegúrate de tener un Mac con **Apple Silicon**. Borra la carpeta `~/.darhisper_config.json` para resetear la config. |
| **La primera transcripción tarda** | Es normal. La primera vez, la app descarga los modelos de IA (1-3 GB). Las siguientes serán instantáneas. |

---

## 📄 Licencia

Este proyecto es de código abierto. Siéntete libre de modificarlo, mejorarlo y compartirlo.

---
*Hecho para maximizar tu productividad.*
