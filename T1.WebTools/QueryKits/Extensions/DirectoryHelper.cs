using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace QueryKits.Extensions
{
    public static class DirectoryHelper
    {
        public static void EnsureDirectory(string directory)
        {
            if (Directory.Exists(directory))
            {
                return;
            }

            Directory.CreateDirectory(directory);
        }
    }
}
