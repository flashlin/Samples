using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace Generated
{
    public class AgentMappingEntityConfiguration : IEntityTypeConfiguration<AgentMappingEntity>
    {
        public void Configure(EntityTypeBuilder<AgentMappingEntity> builder)
        {
            builder.ToTable("AgentMapping");


            builder.Property(x => x.ISOCurrency)
                .HasColumnType("char(3)")
                .IsRequired()
                .HasMaxLength(3)
            ;

            builder.Property(x => x.AgentName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.SubAgentName)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.Enable)
                .HasColumnType("bit")
                .IsRequired()
            ;

            builder.Property(x => x.ModifyBy)
                .HasColumnType("nvarchar(50)")
                .IsRequired()
                .HasMaxLength(50)
            ;

            builder.Property(x => x.ModifyOn)
                .HasColumnType("datetime")
                .IsRequired()
            ;

        }
    }
}
