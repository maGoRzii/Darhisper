# Ghost Eagle 🦅🎙️

Aplicación de barra de menú para macOS que transcribe voz a texto localmente y a ultra-velocidad usando `mlx-whisper` (optimizado para Apple Silicon). También soporta transcripción en la nube vía Google Gemini.

## 🍎 Requisitos

*   **Mac con Apple Silicon** (Chip M1, M2, M3, etc.). **NO funciona en procesadores Intel.**
*   macOS 12.0 o superior.
*   Conexión a internet (solo para la primera descarga de modelos).

## 📥 Instalación (Paso a Paso)

Si tienes el archivo `GhostEagle.app`, sigue estos pasos para instalarlo en un Mac nuevo:

1.  **Copiar la App**:
    Arrastra el archivo `GhostEagle.app` a la carpeta de **Aplicaciones** de tu Mac.

2.  **Permitir Ejecución (Gatekeeper)**:
    Como la app no está firmada por Apple, macOS podría bloquearla al principio. Para solucionarlo:
    *   Abre la **Terminal** (Comando + Espacio -> escribe "Terminal").
    *   Copia y pega este comando y pulsa Enter:
        ```bash
        xattr -cr /Applications/GhostEagle.app
        ```
    *(Esto elimina la marca de cuarentena que macOS pone a las apps descargadas de internet/airdrop)*.

3.  **Primer Inicio y Modelos**:
    *   Abre **GhostEagle** desde tu carpeta de Aplicaciones.
    *   Verás un icono de micrófono 🎙️ en la barra de menú superior.
    *   **¡Paciencia!** La primera vez que intentes transcribir, la app parecerá congelada unos segundos/minutos. Está descargando los modelos de IA en segundo plano.

4.  **Permisos de macOS**:
    El sistema te pedirá permisos la primera vez. Es CRÍTICO que aceptes todos para que funcione:
    *   🎤 **Micrófono**: Para escucharte.
    *   ⌨️ **Accesibilidad**: Para detectar cuando presionas el atajo de teclado y pegar el texto.
    *   🤖 **Eventos del Sistema**: Para controlar el teclado virtual.

## 🎙️ Uso

1.  **Transcribir**:
    *   Coloca el cursor donde quieras escribir (Word, Notas, Slack...).
    *   Mantén pulsado el atajo (Por defecto **F5** o **Option Derecho**).
    *   Escucharás un *beep* y verás una onda de voz en pantalla. Habla.
    *   Suelta la tecla. El texto se escribirá automáticamente.

2.  **Configuración**:
    Haz clic en el icono 🎙️ de la barra de menú para:
    *   **Model**: Cambiar entre modelos locales (MLX) o nube (Gemini).
    *   **Shortcut**: Elegir o grabar un nuevo atajo de teclado.
    *   **API Keys**: Configurar tu clave de Gemini si usas modelos en la nube.

---

## 🛠️ Desarrollo (Para Programadores)

Si quieres ejecutar el código fuente o compilar tu propia versión:

1.  **Clonar e Instalar**:
    ```bash
    git clone https://github.com/maGoRzii/Darhisper.git
    cd Darhisper
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Compilar .app**:
    ```bash
    python3 setup.py py2app
    ```
    La aplicación se generará en la carpeta `dist/`.
