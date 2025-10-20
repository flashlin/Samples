using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class DisplayNamePrefixEntityConfiguration : IEntityTypeConfiguration<DisplayNamePrefixEntity>
    {
        public void Configure(EntityTypeBuilder<DisplayNamePrefixEntity> builder)
        {
            builder.ToTable("DisplayNamePrefix");

            builder.HasKey(x => x.Id);

            builder.Property(x => x.Id)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.PrefixName)
                .HasColumnType("nvarchar(20)")
                .IsRequired()
                .HasMaxLength(20)
            ;

        }
    }
}
