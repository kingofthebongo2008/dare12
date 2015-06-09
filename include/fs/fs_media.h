#pragma once

#include <string>

namespace fs
{
    class media_source
    {
    public:

        media_source(const wchar_t* file_name) : m_path(file_name)
        {

        }

        media_source(const std::wstring& file_name) : m_path(file_name)
        {

        }

        media_source(std::wstring&& file_name) : m_path(std::move(file_name))
        {

        }

        const wchar_t* get_path() const
        {
            return m_path.c_str();
        }

    private:

        std::wstring m_path;
    };

    class media_url
    {
    public:

        media_url(const wchar_t* file_name) : m_file_name(file_name)
        {

        }

        media_url(const std::wstring& file_name) : m_file_name(file_name)
        {

        }

        media_url(std::wstring&& file_name) : m_file_name(std::move(file_name))
        {

        }

        const wchar_t* get_path() const
        {
            return m_file_name.c_str();
        }

    private:

        std::wstring m_file_name;
    };

    inline media_url build_media_url(const media_source& source, const wchar_t* path)
    {
        return std::move(media_url(std::move(std::wstring(source.get_path()) + std::wstring(path))));
    }
}
