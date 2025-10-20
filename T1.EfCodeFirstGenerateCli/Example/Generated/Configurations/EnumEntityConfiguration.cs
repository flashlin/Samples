using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class EnumEntityConfiguration : IEntityTypeConfiguration<EnumEntity>
    {
        public void Configure(EntityTypeBuilder<EnumEntity> builder)
        {
            builder.ToTable("Enum");

            builder.HasKey(x => new { x.Name, x.Option, x.DBValue });

            builder.Property(x => x.Name)
                .HasColumnType("varchar(50)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Option)
                .HasColumnType("varchar(50)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.DBValue)
                .HasColumnType("varchar(50)")
                .ValueGeneratedOnAdd()
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.AppValue)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.AppSource)
                .HasColumnType("varchar(100)")
                .HasMaxLength(100)
            ;

            builder.Property(x => x.AppliedTable)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.AppliedColumn)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Remark)
                .HasColumnType("varchar(200)")
                .HasMaxLength(200)
            ;

        }
    }
}
