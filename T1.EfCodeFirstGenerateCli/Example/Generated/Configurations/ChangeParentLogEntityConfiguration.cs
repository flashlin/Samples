using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class ChangeParentLogEntityConfiguration : IEntityTypeConfiguration<ChangeParentLogEntity>
    {
        public void Configure(EntityTypeBuilder<ChangeParentLogEntity> builder)
        {
            builder.ToTable("ChangeParentLog");

            builder.HasKey(x => x.LogID);

            builder.Property(x => x.LogID)
                .HasColumnType("int(10,0)")
                .ValueGeneratedOnAdd()
                .IsRequired()
            ;

            builder.Property(x => x.UserName)
                .HasColumnType("varchar(250)")
                .HasMaxLength(250)
            ;

            builder.Property(x => x.OldParentName)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.NewParentName)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifiedDate)
                .HasColumnType("datetime")
            ;

            builder.Property(x => x.ModifiedBy)
                .HasColumnType("varchar(50)")
                .HasMaxLength(50)
            ;

        }
    }
}
